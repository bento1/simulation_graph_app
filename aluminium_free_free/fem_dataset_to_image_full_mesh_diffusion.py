# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
from utils import feature_normalize,minmax_scale,standard_scale
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Data, Dataset
from tqdm import tqdm
# from torch_geometric.utils import k_hop_subgraph
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from torch.utils.data import DataLoader
from copy import copy
def csvToImage(df,GRID=256):
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    z_min, z_max = df["z"].min(), df["z"].max()
    points = df[["x", "y", "z"]].values          # (N,3)
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(x_min, x_max, GRID),
        np.linspace(y_min, y_max, GRID),
        np.linspace(z_min, z_max, GRID),
        indexing="ij"
    )
    values_uz = df["uz"].values       
    Ux_grid = griddata(
        points,
        values_uz,
        (grid_x, grid_y, grid_z),
        method="linear",
        fill_value=0.0
    ).astype(np.float32)
    
    return Ux_grid



def csvToImage_25d_slice(
    df,
    z_idx,
    GRID,
    scale,
):
    """
    df     : dataframe with columns [x,y,z,uz]
    z_idx  : slice index (0 ~ GRID-1)
    GRID   : output resolution
    scale  : global scale dict
    return : (GRID, GRID) float32
    """

    # -----------------------------
    # global z grid (고정)
    # -----------------------------
    xmin=df['x'].min()
    xmax=df['x'].max()
    ymin=df['y'].min()
    ymax=df['y'].max()
    zmin=df['z'].min()
    zmax=df['z'].max()
    Lz=scale['Lz']
    nz=scale['nz']
    z_lin = np.linspace(zmin, zmax, GRID)
    z0 = z_lin[z_idx]

    dz = (zmax - zmin) / (GRID - 1)
    z_low  = z0 -  Lz/nz/2
    z_high = z0 +  Lz/nz/2

    # -----------------------------
    # z-band 필터
    # -----------------------------
    sdf = df[(df["z"] >= z_low) & (df["z"] <= z_high)]

    if len(sdf) < 5:
        print('no data')
        return np.zeros((GRID, GRID), dtype=np.float32)

    # -----------------------------
    # global x,y grid (고정)
    # -----------------------------
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, GRID),
        np.linspace(ymin, ymax, GRID),
        indexing="ij"
    )

    # -----------------------------
    # 2D interpolation
    # -----------------------------
    img = griddata(
        sdf[["x", "y"]].values,
        sdf["uz"].values,
        (grid_x, grid_y),
        method="nearest",   # ← 안정 + 빠름
    ).astype(np.float32)

    return img

def csvToImage_25d_stack(
    df,
    z_idx,
    K,
    GRID,
    scale
):
    """
    return: (2K+1, GRID, GRID)
    """
    slices = []
    for dz in range(-K, K + 1):
        zi = np.clip(z_idx + dz, 0, GRID - 1)
        img = csvToImage_25d_slice(
            df,
            zi,
            GRID,
            scale
        )
        slices.append(img)

    return np.stack(slices, axis=0)


def make_25d_xy_from_xyz(vol, z_idx, K=2):
    """
    vol   : (X, Y, Z) torch.Tensor
    z_idx : center z index
    K     : number of neighbor slices
    return: (2K+1, X, Y)
    """
    X, Y, Z = vol.shape

    idx = torch.arange(z_idx - K, z_idx + K + 1, device=vol.device)
    idx = idx.clamp(0, Z - 1)

    # 슬라이스: (X, Y, 2K+1)
    slices = vol[:, :, idx]

    # 채널 우선으로 이동: (2K+1, X, Y)
    slices = slices.permute(2, 0, 1)

    return slices


def build_node_features(nodes_xyz: np.ndarray, params: dict) -> np.ndarray:
    N = nodes_xyz.shape[0]
    param_list=[params[k] for k in sorted(params.keys()) ]

    xyz = nodes_xyz.astype(np.float32)

    global_vec = np.array(param_list, dtype=np.float32)  # 6
    global_feat = np.repeat(global_vec[None, :], N, axis=0)

    x = np.concatenate([xyz, global_feat], axis=1)  # [N, 12]
    return x

class FemImageDataset(Dataset):
    def __init__(self, root_dir: str,type='train',GRID=256,in_channels=5):
        super().__init__()
        self.root_dir = root_dir
        self.samples = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.data_list = []
        self.scale_info={}
        self.type=type
        self.GRID=GRID
        self.in_channels=in_channels
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
    
    def _load_image(self, sd):
        
        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        # disp_df=copy(origin_disp_df)
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            origin_params = json.load(f)
        params,Lx, Ly, Lz = feature_normalize(copy(origin_params),self.scale_info)
        for key in ["ux","uy","uz"]:
            disp_df[key]= disp_df[key].apply(lambda v:standard_scale(v,self.scale_info[key]['std'],self.scale_info[key]['mean'],self.scale_info[key]['max']))
        vols=[] 
        params_lists=[] 
        for z_idx in range(self.GRID): 
            # z_idx = np.random.randint(0, self.GRID) 
            K=int((self.in_channels-1)/2) 
            image=csvToImage_25d_stack(disp_df, z_idx, K=K,GRID=self.GRID,scale=origin_params) 
            # image: (W,H,D,C) numpy 
            vol = torch.from_numpy(image).float() 
            vols.append(vol) 
            params_list=[params[k] for k in sorted(params)] 
            params_list=[z_idx/self.GRID]+params_list 
            params_lists.append(params_list) 

        data={ "image":vols, "cond":params_lists} # "slice_z_idx": torch.tensor([z_idx]), }


        return data
        
    def __getitem__(self, idx):
        return self._load_image(self.samples[idx])
    
class FemImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, scale_info:dict ,GRID=256,in_channels=5):
        super().__init__()
        self.root_dir = root_dir
        self.samples = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.data_list = []
        self.GRID=GRID
        self.in_channels=in_channels
        self.scale_info=scale_info

        print("Loading FEM graph dataset...")
        for sd in tqdm(self.samples):
            full_data = self._load_image(sd)
            self.data_list.append(full_data)
        print("Build Complete FEM graph dataset...")
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, sd):

        disp_df  = pd.read_csv(os.path.join(sd, "nodal_stress_disp.csv")).sort_values("node_id")
        # disp_df=copy(origin_disp_df)
        with open(os.path.join(sd, "params.json"), "r", encoding="utf-8") as f:
            origin_params = json.load(f)
        params,Lx, Ly, Lz = feature_normalize(copy(origin_params),self.scale_info)
        for key in ["ux","uy","uz"]:
            disp_df[key]= disp_df[key].apply(lambda v:standard_scale(v,self.scale_info[key]['std'],self.scale_info[key]['mean'],self.scale_info[key]['max']))
        z_idx = np.random.randint(0, self.GRID)
        # K=int((self.in_channels-1)/2)
        # image=csvToImage_25d_stack(disp_df, z_idx, K=K,GRID=self.GRID,scale=origin_params)

        # image: (W,H,D,C) numpy
        # vol = torch.from_numpy(image).float()
        result=[]
        for z_idx in range(self.GRID):
            params_list=[params[k] for k in sorted(params)]
            params_list=[z_idx/self.GRID]+params_list
            data={
                # "image":vol,
                "cond":torch.Tensor(params_list),
                "slice_z_idx": torch.tensor([z_idx]),
            }
            result.append(data)
        return result
        
    def __getitem__(self, idx):
        return self._load_image(self.samples[idx])
    
if __name__ == "__main__":
    root = "./data"
    ds = FemImageDataset(root_dir=root,type='train',GRID=256)
    train_loader = DataLoader(ds, batch_size=2, shuffle=True,)
    for i, batch in enumerate(train_loader):
        print(i)