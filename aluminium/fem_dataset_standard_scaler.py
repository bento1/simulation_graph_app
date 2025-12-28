# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
from utils import feature_standard_normalize,standard_scale
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

def cells_to_edges(cells_df: pd.DataFrame, undirected=True) -> np.ndarray:
    cols = ["node_0", "node_1", "node_2", "node_3"]
    tets = cells_df[cols].to_numpy(dtype=np.int64)

    edges = []
    for a, b, c, d in tets:
        pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edges.extend(pairs)
        if undirected:
            edges.extend([(j,i) for (i,j) in pairs])

    edges = np.array(edges, dtype=np.int64)
    edges = np.unique(edges, axis=0)
    return edges  # [E,2]

def knn_edges(xyz: np.ndarray, k: int = 12, undirected=True) -> np.ndarray:
    """
    xyz: [N,3]
    O(N^2)라서 N이 아주 크면 느림. (N이 커지면 torch_cluster.knn_graph 쓰는게 정석)
    """
    N = xyz.shape[0]
    # pairwise dist^2
    d2 = np.sum((xyz[:, None, :] - xyz[None, :, :])**2, axis=2)  # [N,N]
    np.fill_diagonal(d2, np.inf)

    edges = []
    for i in range(N):
        nn_idx = np.argpartition(d2[i], kth=min(k, N-1)-1)[:min(k, N-1)]
        for j in nn_idx:
            edges.append((i, int(j)))
            if undirected:
                edges.append((int(j), i))
    edges = np.array(edges, dtype=np.int64)
    edges = np.unique(edges, axis=0)
    return edges  # [E,2]


def build_edge_attr(edge_index: torch.Tensor, xyz: torch.Tensor, Lx, Ly, Lz) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    dvec = xyz[dst] - xyz[src]                # [E,3]
    L=torch.norm(torch.tensor([float(Lx), float(Ly), float(Lz)]))
    dist = torch.norm(dvec, dim=1, keepdim=True) + 1e-12  # [E,1]
    edge_attr = torch.cat([dvec, dist], dim=1)/L            # [E,4]
    return edge_attr


def build_node_features(nodes_xyz: np.ndarray, params: dict) -> np.ndarray:
    N = nodes_xyz.shape[0]
    param_list=[params[k] for k in sorted(params.keys()) ]

    xyz = nodes_xyz.astype(np.float32)

    global_vec = np.array(param_list, dtype=np.float32)  # 6
    global_feat = np.repeat(global_vec[None, :], N, axis=0)

    x = np.concatenate([xyz, global_feat], axis=1)  # [N, 12]
    return x

class FemGraphDataset(Dataset):
    def __init__(self, root_dir: str, knn_k: int = 12, use_cell_edges: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.knn_k = knn_k
        self.use_cell_edges = use_cell_edges
        self.samples = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.data_list = []
        self.scale_info={}
        for sd in tqdm(self.samples):
            try:
                self.find_data_info(sd)
            except Exception as e:
                print(f"Error in finding data info for {sd}: {e}")
                continue
        print("Loading FEM graph dataset...")
        for sd in tqdm(self.samples):
            try:
                full_data = self._load_full_graph(sd)
                subgraphs = self._split_graph(full_data, 100)
                self.data_list.extend(subgraphs)
            except Exception as e:
                print(f"Error in loading graph for {sd}: {e}")
                continue
            # break
        print("Build Complete FEM graph dataset...")
    
    def find_data_info(self,sd):
        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        for key in ["x","y","z","sigma_xx","sigma_yy","sigma_zz","tau_xy","tau_yz","tau_zx","ux","uy","uz"]:
            mean_value=disp_df[key].mean()
            std_value=disp_df[key].std()
            num_data=disp_df[key].shape[0]
            min_value=disp_df[key].min()
            max_value=disp_df[key].max()
            if key not in self.scale_info:
                self.scale_info[key]={'min':min_value,'max':max_value,'mean':mean_value,'std':std_value,'num':num_data}
            else:
                if min_value<=self.scale_info[key]['min']:
                    self.scale_info[key]['min']=min_value
                if max_value>self.scale_info[key]['max']:
                    self.scale_info[key]['max']=max_value
                prev_mean=self.scale_info[key]['mean']
                prev_num=self.scale_info[key]['num']
                prev_std=self.scale_info[key]['std']
                self.scale_info[key]['mean']= (mean_value*num_data+ self.scale_info[key]['mean']*self.scale_info[key]['num'])/(num_data+self.scale_info[key]['num'])
                self.scale_info[key]['std']=np.sqrt(((num_data-1)*std_value**2 + \
                                                    (prev_num-1)*prev_std**2+ \
                                                    num_data*(mean_value-self.scale_info[key]['mean'])**2+ \
                                                    prev_num*(prev_mean-self.scale_info[key]['mean'])**2)/ (num_data +prev_num -1))
                self.scale_info[key]['num']+=num_data

        for key in params:
            num_data=1
            if key not in self.scale_info:
                self.scale_info[key]={'min':params[key],'max':params[key],'mean':params[key],'std':0,'num':1}
            else:
                if params[key]<self.scale_info[key]['min']:
                    self.scale_info[key]['min']=params[key]
                if params[key]>self.scale_info[key]['max']:
                    self.scale_info[key]['max']=params[key]            
                prev_mean=self.scale_info[key]['mean']
                prev_num=self.scale_info[key]['num']
                prev_std=self.scale_info[key]['std']
                self.scale_info[key]['mean']= prev_mean+(params[key]-prev_mean)/(prev_num+1)
                del1=params[key]-prev_mean
                del2=params[key]-self.scale_info[key]['mean']
                M2=prev_std**2 * (prev_num -1) + del1 * del2
                self.scale_info[key]['std']=np.sqrt(M2/(self.scale_info[key]['num']-1)) if self.scale_info[key]['num']>1 else 0
                self.scale_info[key]['num']+=num_data
    
    def len(self):
        return len(self.data_list)
    
    def _split_graph(self, data: Data, max_nodes=2000):
        subgraphs = []
        N = data.num_nodes
        perm = torch.randperm(N)

        for i in range(0, N, max_nodes):
            idx = perm[i:i+max_nodes]
            sub = data.subgraph(idx)
            subgraphs.append(sub)

        return subgraphs

    def _load_full_graph(self, sd):


        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        params,Lx, Ly, Lz = feature_standard_normalize(params,self.scale_info)
        for key in ["x","y","z","ux","uy","uz"]:
            disp_df[key]= disp_df[key].apply(lambda v:standard_scale(v,self.scale_info[key]['std'],self.scale_info[key]['mean'],self.scale_info[key]['max']))
        xyz = disp_df[["x","y","z"]].to_numpy(dtype=np.float32)
        y = disp_df[["ux","uy","uz"]].to_numpy(dtype=np.float32)

        x = build_node_features(xyz, params)  # [N,F]

        # edges: cell + knn 혼합
        edges = []
        if self.use_cell_edges and os.path.exists(os.path.join(sd, "edge_infos.csv")):
            cells_df = pd.read_csv(os.path.join(sd, "edge_infos.csv"))
            edges.append(cells_to_edges(cells_df, undirected=True))
        # edges.append(knn_edges(xyz, k=self.knn_k, undirected=True))

        edges = np.vstack(edges)
        edges = np.unique(edges, axis=0)

        edge_index = torch.from_numpy(edges).t().contiguous()  # [2,E]

        data = Data(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            pos=torch.from_numpy(xyz),   # pos는 따로 보관(편함)
            edge_index=edge_index
        )

        # edge_attr 생성
        data.edge_attr = build_edge_attr(data.edge_index, data.pos,Lx, Ly, Lz)
        return data
        
    def get(self, idx):
        return self.data_list[idx]
class FemGraphInferenceDataset(Dataset):
    def __init__(self, root_dir: str, scale_info:dict, knn_k: int = 12, use_cell_edges: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.knn_k = knn_k
        self.use_cell_edges = use_cell_edges
        self.samples = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.data_list = []
        self.scale_info=scale_info

        print("Loading FEM graph dataset...")
        for sd in tqdm(self.samples):
            full_data = self._load_full_graph(sd)
            self.data_list.append(full_data)
        print("Build Complete FEM graph dataset...")
    
    def len(self):
        return len(self.data_list)
    
    def _load_full_graph(self, sd):

        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        params,Lx, Ly, Lz = feature_standard_normalize(params,self.scale_info)
        for key in ["x","y","z",]:
            disp_df[key]= disp_df[key].apply(lambda v:standard_scale(v,self.scale_info[key]['std'],self.scale_info[key]['mean'],self.scale_info[key]['max']))
        xyz = disp_df[["x","y","z"]].to_numpy(dtype=np.float32)

        x = build_node_features(xyz, params)  # [N,F]

        # edges: cell + knn 혼합
        edges = []
        if self.use_cell_edges and os.path.exists(os.path.join(sd, "edge_infos.csv")):
            cells_df = pd.read_csv(os.path.join(sd, "edge_infos.csv"))
            edges.append(cells_to_edges(cells_df, undirected=True))
        # edges.append(knn_edges(xyz, k=self.knn_k, undirected=True))

        edges = np.vstack(edges)
        edges = np.unique(edges, axis=0)

        edge_index = torch.from_numpy(edges).t().contiguous()  # [2,E]

        data = Data(
            x=torch.from_numpy(x),
            #y=torch.from_numpy(y),
            pos=torch.from_numpy(xyz),   # pos는 따로 보관(편함)
            edge_index=edge_index
        )

        # edge_attr 생성
        data.edge_attr = build_edge_attr(data.edge_index, data.pos,Lx, Ly, Lz)
        return data
        
    def get(self, idx):
        return self.data_list[idx]
if __name__ == "__main__":
    root = "./data"
    ds = FemGraphDataset(root_dir=root, knn_k=12, use_cell_edges=True)