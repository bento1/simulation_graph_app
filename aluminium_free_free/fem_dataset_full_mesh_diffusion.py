# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
from utils import feature_normalize,minmax_scale
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Data, Dataset
from tqdm import tqdm
# from torch_geometric.utils import k_hop_subgraph

def build_node_features(nodes_xyz: np.ndarray, params: dict) -> np.ndarray:
    N = nodes_xyz.shape[0]
    param_list=[params[k] for k in sorted(params.keys()) ]

    xyz = nodes_xyz.astype(np.float32)

    global_vec = np.array(param_list, dtype=np.float32)  # 6
    global_feat = np.repeat(global_vec[None, :], N, axis=0)

    x = np.concatenate([xyz, global_feat], axis=1)  # [N, 12]
    return x

class FemGraphDataset(Dataset):
    def __init__(self, root_dir: str,type='train'):
        super().__init__()
        self.root_dir = root_dir
        self.samples = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.data_list = []
        self.scale_info={}
        self.type=type
        for sd in tqdm(self.samples):
            try:
                self.find_data_info(sd)
            except Exception as e:
                print(f"Error in finding data info for {sd}: {e}")
                continue
        print("Loading FEM graph dataset...")
        cut_ind=int(len(self.samples)*0.9)
        if self.type=='train':
            self.samples=self.samples[:cut_ind]
        else:
            self.samples=self.samples[cut_ind:]
        print("Build Complete FEM graph dataset...")
    
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
    
    def __len__(self):
        return len(self.samples)
    

    def _load_full_graph(self, sd):


        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        params,Lx, Ly, Lz = feature_normalize(params,self.scale_info)
        for key in ["x","y","z","ux","uy","uz"]:
            disp_df[key]= disp_df[key].apply(lambda v:minmax_scale(v,self.scale_info[key]['min'],self.scale_info[key]['max']))
        xyz = disp_df[["x","y","z"]].to_numpy(dtype=np.float32)
        y = disp_df[["uz"]].to_numpy(dtype=np.float32)

        x = build_node_features(xyz, params)  # [N,F]
        data={
            "pos":torch.Tensor(xyz),
            "cond":torch.Tensor(x),
            "x0":torch.Tensor(y)
        }


        return data
        
    def __getitem__(self, idx):
        return self._load_full_graph(self.samples[idx])
    
class FemGraphInferenceDataset(Dataset):
    def __init__(self, root_dir: str, scale_info:dict, ):
        super().__init__()
        self.root_dir = root_dir
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
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_full_graph(self, sd):

        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
        params,Lx, Ly, Lz = feature_normalize(params,self.scale_info)
        for key in ["x","y","z",]:
            disp_df[key]= disp_df[key].apply(lambda v:minmax_scale(v,self.scale_info[key]['min'],self.scale_info[key]['max']))
        xyz = disp_df[["x","y","z"]].to_numpy(dtype=np.float32)

        x = build_node_features(xyz, params)  # [N,F]

        data={
            "pos":torch.Tensor(xyz),
            "cond":torch.Tensor(x),
        }


        return data
        
    def __getitem__(self, idx):
        return self._load_full_graph(self.samples[idx])
if __name__ == "__main__":
    root = "./data"
    ds = FemGraphDataset(root_dir=root, knn_k=12, use_cell_edges=True)