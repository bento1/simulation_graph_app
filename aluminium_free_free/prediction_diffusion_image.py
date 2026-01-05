
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fem_model import ContextUnet
from tqdm import tqdm
import json
from utils import draw_disp_on_mesh_3d,draw_mesh,inverse_minmax_scale
import pandas as pd
from fem_dataset_to_image_full_mesh_diffusion import FemImageInferenceDataset as MeshDataset
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def p_sample(model, x_t, t, cond, sched):
    """
    x_t  : (B,C,H,W)
    t    : int scalar
    cond : (B,cond_dim)
    """
    B = x_t.shape[0]

    beta_t  = sched.betas[t]
    alpha_t = sched.alphas[t]
    ab_t    = sched.alpha_bar[t]

    # t를 model 입력 형태로
    t_tensor = torch.full((B,), t, device=x_t.device)
    t_norm = t_tensor / sched.T

    # ε 예측
    eps_pred = model(x_t, t_norm, cond)

    # 평균 계산
    mean = (
        (1.0 / torch.sqrt(alpha_t)) *
        (x_t - ((1 - alpha_t) / torch.sqrt(1 - ab_t)) * eps_pred)
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise
    else:
        x_prev = mean  # t=0이면 noise 없음

    return x_prev


@torch.no_grad()
def sample_image(
    model,
    cond,
    sched,
    shape,
    device="cuda",
):
    """
    shape : (B,C,H,W)
    cond  : (B,cond_dim)
    """
    model.eval()

    # x_T ~ N(0, I)
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(sched.T)):
        x_t = p_sample(model, x_t, t, cond, sched)

    return x_t


class DiffusionScheduler:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

def main():
    root = "./test"
    if torch.cuda.is_available() :
        device='cuda'
    elif torch.backends.mps.is_available() :
        device='mps'
    else :
        device='cpu'
    file_name="diffustion_ver3_early"
    with open(f"model_param_{file_name}.json", "r", encoding="utf-8") as f:
        model_param = json.load(f)

    ds = MeshDataset(root_dir=root,scale_info=model_param['dataset_scale_info'],GRID=model_param['GRID'],in_channels=model_param['in_channels'])
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = ContextUnet(
        in_channels=model_param['in_channels'],
        out_channels=model_param['out_channels'],
        n_feat=model_param['n_feat'],
        n_cfeat=model_param['n_cfeat'],
        height=model_param['GRID'],
        ).to(device)

    checkpoint = torch.load(f"mesh_invariant_{file_name}.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    sched = DiffusionScheduler(device=device)
    sched.T=1000
    K=int((model_param['in_channels']-1)/2)
    for batch in loader:
        model.eval()

        # batch = {k: v.to(device) for k, v in batch.items()}
        data=[]
        for m in batch:
            pos  = m["slice_z_idx"]
            cond = m["cond"].to(device)
            x_hat = sample_image(model, cond, sched,(1,model_param['in_channels'],model_param['GRID'],model_param['GRID']), device)
            data.append(x_hat.cpu().detach().numpy()[0,K,:,:])


    for i in range(data.shape[0]):
        data[i,:]=inverse_minmax_scale(data[i,:],model_param['dataset_scale_info']['uz']['min'],model_param['dataset_scale_info']['uz']['max'])
    # {
    #   "Lx": 6,
    #   "Ly": 2,
    #   "Lz": 2,
    #   "nx": 16,
    #   "ny": 18,
    #   "nz": 10,
    #   "xm0": 0.9972,
    #   "xm1": 1.7471999999999999,
    #   "ym0": 0.36,
    #   "ym1": 0.5822222222222222,
    #   "zm0": 0,
    #   "zm1": 0.4,
    #   "E": 69000000000.0,
    #   "nu": 0.33,
    #   "rho": 2700,
    #   "m_add": 31,
    #   "freq": 262,
    #   "a_base": 15,
    #   "zeta": 0.005
    # }
    with open(f"params.json", "r", encoding="utf-8") as f:
        params = json.load(f)
    x_lin = np.linspace(0, params['Lx'], model_param['GRID'])
    y_lin = np.linspace(0, params['Ly'], model_param['GRID'])
    z_lin = np.linspace(0, params['Lz'], model_param['GRID'])
    draw_disp_on_mesh_3d(x_lin,y_lin,z_lin,data)

if __name__=='__main__':
    main()