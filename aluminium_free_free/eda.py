# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
from utils import feature_standard_normalize,standard_scale,minmax_scale
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
import matplotlib.pyplot as plt

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
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

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
        self.max_leng=0
        for sd in tqdm(self.samples):
            try:
                self.find_full_data(sd)
            except Exception as e:
                print(f"Error in finding data info for {sd}: {e}")
                continue

        plt.figure(0)
        data=pd.concat(self.data_list)
        for g, sub in data.groupby("name"):
            plt.hist(sub["uz"], bins=10, alpha=0.6, label=g)
        plt.xlabel("signed log(value)")
        plt.show()
        plt.figure(0)   
        for g, sub in data.groupby("name"):
            plt.hist(signed_log(sub["uz"]), bins=50, alpha=0.5, label=g)
        plt.xlabel("signed log(value)")
        plt.show()

        center = data["uz"].abs() < 1e-9   # 기준은 데이터 보고 조절
        plt.figure(0) 
        for g, sub in data[center].groupby("name"):
            plt.hist(sub["uz"], bins=50, alpha=0.6, label=g)

        plt.title("Zoomed near mean")
        plt.show()
        plt.figure(figsize=(8, 4))
        data.boxplot(column="uz", by="name", showfliers=True)
        plt.title("Boxplot by Group")
        plt.suptitle("")  # pandas 기본 제목 제거
        plt.ylabel("value")
        plt.show()
    def find_data_info(self,sd):
        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        for key in ["x","y","z","ux","uy","uz","ux_abs","uy_abs","uz_abs"]:
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
    def find_full_data(self,sd):
        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")

        for key in ["uz"]:
            mean_value=disp_df[key].values.tolist()
            mean_value=pd.DataFrame(mean_value,columns=[key])
            mean_value['name']=sd
            self.data_list.append(mean_value)

    
    def len(self):
        return len(self.data_list)
    

    def _load_full_graph(self, sd):


        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        params,Lx, Ly, Lz = feature_standard_normalize(params,self.scale_info)
        for key in ["x","y","z","ux","uy","uz"]:
            disp_df[key]= disp_df[key].apply(lambda v:minmax_scale(v,self.scale_info[key]['min'],self.scale_info[key]['max']))
        vmin=disp_df["uz"].min()
        vmax=disp_df["uz"].max()
        vmean=disp_df["uz"].mean()
        vstd=disp_df["uz"].std()

        return {"vmin":vmin,"vmax":vmax,"vmean":vmean,"vstd":vstd}
        
    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    root = "./data"
    ds = FemGraphDataset(root_dir=root, knn_k=12, use_cell_edges=True)