import os

def isExistFolder(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def makeFolder(folder_path):
    if not isExistFolder(folder_path):
        os.makedirs(folder_path)
        return True
    return False
def minmax_scale(x, xmin, xmax, s=(0,1)):
    """
    x    : input tensor / scalar
    xmin : min value
    xmax : max value
    a,b  : target range [a,b]
    """
    a,b=s
    if xmax == xmin:
        # 정보가 없을 때 중앙값으로 보냄
        return (a + b) / 2

    scaled_01 = (x - xmin) / (xmax - xmin)
    scaled_ab = scaled_01 * (b - a) + a
    return scaled_ab

def inverse_minmax_scale(x_scaled, xmin, xmax, s=(0,1)):
    """
    x_scaled : scaled value in [a,b]
    xmin,xmax: original range
    a,b      : scaling range
    """
    a,b=s
    if xmax == xmin:
        return xmin

    scaled_01 = (x_scaled - a) / (b - a)
    x = scaled_01 * (xmax - xmin) + xmin
    return x



def feature_normalize(params,scale_info,s=(0,1)):

    Lx=scale_info['Lx']['max']
    Ly=scale_info['Ly']['max']
    Lz=scale_info['Lz']['max']


    params['Lx']=minmax_scale(params['Lx'],scale_info['Lx']['min'],scale_info['Lx']['max'],s)
    params['Ly']=minmax_scale(params['Ly'],scale_info['Ly']['min'],scale_info['Ly']['max'],s)
    params['Lz']= minmax_scale(params['Lz'],scale_info['Lz']['min'],scale_info['Lz']['max'],s)
    params['nx']=minmax_scale(params['nx'],scale_info['nx']['min'],scale_info['nx']['max'],s)
    params['ny']=minmax_scale(params['ny'],scale_info['ny']['min'],scale_info['ny']['max'],s)
    params['nz']=minmax_scale(params['nz'],scale_info['nz']['min'],scale_info['nz']['max'],s)
    params['xm0']=minmax_scale(params['xm0'],scale_info['xm0']['min'],scale_info['xm0']['max'],s)
    params['xm1']=minmax_scale(params['xm1'],scale_info['xm1']['min'],scale_info['xm1']['max'],s)
    params['ym0']=minmax_scale(params['ym0'],scale_info['ym0']['min'],scale_info['ym0']['max'],s)
    params['ym1']=minmax_scale(params['ym1'],scale_info['ym1']['min'],scale_info['ym1']['max'],s)
    params['zm0']=minmax_scale(params['zm0'],scale_info['zm0']['min'],scale_info['zm0']['max'],s)
    params['zm1']=minmax_scale(params['zm1'],scale_info['zm1']['min'],scale_info['zm1']['max'],s)
    params['E']=minmax_scale(params['E'],scale_info['E']['min'],scale_info['E']['max'],s)
    params['nu']=minmax_scale(params['nu'],scale_info['nu']['min'],scale_info['nu']['max'],s)
    params['rho']=minmax_scale(params['rho'],scale_info['rho']['min'],scale_info['rho']['max'],s)
    params['m_add']=minmax_scale(params['m_add'],scale_info['m_add']['min'],scale_info['m_add']['max'],s)
    params['freq']=minmax_scale(params['freq'],scale_info['freq']['min'],scale_info['freq']['max'],s)
    params['a_base']=minmax_scale(params['a_base'],scale_info['a_base']['min'],scale_info['a_base']['max'],s) 
    params['zeta']=minmax_scale(params['zeta'],scale_info['zeta']['min'],scale_info['zeta']['max'],s)


    return params,Lx, Ly, Lz


def standard_scale(params,__std,__mean,__max):
    if __max==0:
        return 0
    if __std ==0:
        return params/__max/2
    scaled = (params - __mean) / (__std)
    return scaled

def inverse_standard_scale(scaled,__std,__mean,__max):
    if __max==0:
        return 0
    if __std ==0:
        return params*__max*2
    params = scaled*__std +__mean
    return params


def feature_standard_normalize(params,scale_info):

    Lx=scale_info['Lx']['max']
    Ly=scale_info['Ly']['max']
    Lz=scale_info['Lz']['max']


    params['Lx']=standard_scale(params['Lx'],scale_info['Lx']['std'],scale_info['Lx']['mean'],scale_info['Lx']['max'])
    params['Ly']=standard_scale(params['Ly'],scale_info['Ly']['std'],scale_info['Ly']['mean'],scale_info['Ly']['max'])
    params['Lz']= standard_scale(params['Lz'],scale_info['Lz']['std'],scale_info['Lz']['mean'],scale_info['Lz']['max'])
    params['nx']=standard_scale(params['nx'],scale_info['nx']['std'],scale_info['nx']['mean'],scale_info['nx']['max'])
    params['ny']=standard_scale(params['ny'],scale_info['ny']['std'],scale_info['ny']['mean'],scale_info['ny']['max'])
    params['nz']=standard_scale(params['nz'],scale_info['nz']['std'],scale_info['nz']['mean'],scale_info['nz']['max'])
    params['xm0']=standard_scale(params['xm0'],scale_info['xm0']['std'],scale_info['xm0']['mean'],scale_info['xm0']['max'])
    params['xm1']=standard_scale(params['xm1'],scale_info['xm1']['std'],scale_info['xm1']['mean'],scale_info['xm1']['max'])
    params['ym0']=standard_scale(params['ym0'],scale_info['ym0']['std'],scale_info['ym0']['mean'],scale_info['ym0']['max'])
    params['ym1']=standard_scale(params['ym1'],scale_info['ym1']['std'],scale_info['ym1']['mean'],scale_info['ym1']['max'])
    params['zm0']=standard_scale(params['zm0'],scale_info['zm0']['std'],scale_info['zm0']['mean'],scale_info['zm0']['max'])
    params['zm1']=standard_scale(params['zm1'],scale_info['zm1']['std'],scale_info['zm1']['mean'],scale_info['zm1']['max'])
    params['E']=standard_scale(params['E'],scale_info['E']['std'],scale_info['E']['mean'],scale_info['E']['max'])
    params['nu']=standard_scale(params['nu'],scale_info['nu']['std'],scale_info['nu']['mean'],scale_info['nu']['max'])
    params['rho']=standard_scale(params['rho'],scale_info['rho']['std'],scale_info['rho']['mean'],scale_info['rho']['max'])
    params['m_add']=standard_scale(params['m_add'],scale_info['m_add']['std'],scale_info['m_add']['mean'],scale_info['m_add']['max'])
    params['freq']=standard_scale(params['freq'],scale_info['freq']['std'],scale_info['freq']['mean'],scale_info['freq']['max'])
    params['a_base']=standard_scale(params['a_base'],scale_info['a_base']['std'],scale_info['a_base']['mean'],scale_info['a_base']['max']) 
    params['zeta']=standard_scale(params['zeta'],scale_info['zeta']['std'],scale_info['zeta']['mean'],scale_info['zeta']['max'])


    return params,Lx, Ly, Lz


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, mode="min"):
        """
        patience : 개선 없이 버틸 epoch 수
        min_delta: 이 값보다 작게 개선되면 '개선 아님'으로 간주
        mode     : "min" (loss), "max" (accuracy 등)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return True  # best 갱신

        improved = (
            metric < self.best_score - self.min_delta
            if self.mode == "min"
            else metric > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
        
import matplotlib.pyplot as plt
import numpy as np

def draw_disp_on_mesh(x,y,z,u_vec,):
    # x = df["x"].values
    # y = df["y"].values
    # z = df["z"].values

    # df_disp = pd.DataFrame({
    #     "node_id": np.arange(coords_nodes.shape[0]),
    #     "ux": u_vec[:,0],
    #     "uy": u_vec[:,1],
    #     "uz": u_vec[:,2],
    # })
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111, projection="3d")
    q1_3 = np.clip(u_vec, np.quantile(u_vec,0.25),  np.quantile(u_vec,0.75))
    sc = ax.scatter(
        x, y, z,
        c=q1_3,
        cmap="jet",
        s=6,
        alpha=0.9
    )

    # -----------------------
    # 축 비율 현실적으로 맞추기
    # -----------------------
    ax.set_box_aspect([
        np.ptp(x),
        np.ptp(y),
        np.ptp(z)
    ])

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Nodal Displacement Z axis (Base Excitation)")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("displacement [m]")
    # ax.set_box_aspect([Lx,Ly, Lz])  # 비율 유지
    # plt.tight_layout()
    plt.show()

def draw_mesh(x,y,z,Lx,Ly, Lz,nx,ny,nz,xm0,xm1,ym0,ym1,zm0,zm1):

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111, projection="3d")
    mask = (
        (x >= Lx/2-Lx/nx) & (x <= Lx/2+Lx/nx) &
        (y >= Ly/2-Ly/nx) & (y <= Ly/2+Ly/ny) &
        (z >= -Lz/nz) & (z <= Lz/nz)
    )
    # 전체 메시
    ax.scatter(x, y, z, s=1, alpha=0.2, label="mesh")
    ax.scatter(x[mask], y[mask], z[mask], s=10, c="r", label="excitation patch")
    # 질량 블록 판별
    mask = (
        (x >= xm0) & (x <= xm1) &
        (y >= ym0) & (y <= ym1) &
        (z >= zm0) & (z <= zm1)
    )
    ax.scatter(x[mask], y[mask], z[mask], s=10, c="g", label="mass block")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
        # -----------------------
    # 축 비율 현실적으로 맞추기
    # -----------------------
    # ax.set_box_aspect([
    #     np.ptp(x),
    #     np.ptp(y),
    #     np.ptp(z)
    # ])
    ax.set_box_aspect([Lx,Ly, Lz])  # 비율 유지
    plt.show()

from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import save_image, make_grid
import torch 

def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax,(2,3)) 
    xmin = np.expand_dims(xmin,(2,3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)

def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani


import numpy as np
import matplotlib.pyplot as plt

def draw_disp_on_mesh_3d(
    x_lin,
    y_lin,
    z_lin,
    data,
    stride=1,          # 2~4 주면 훨씬 가벼워짐
    clip_q=(0.25,0.75),
    cmap="jet",
):
    """
    x_lin : (nx,)
    y_lin : (ny,)
    z_lin : (nz,)
    data  : (nx, ny, nz)
    """

    # ---------------------------
    # meshgrid (물리 좌표)
    # ---------------------------
    X, Y, Z = np.meshgrid(
        x_lin, y_lin, z_lin,
        indexing="ij"
    )

    # ---------------------------
    # flatten + downsample
    # ---------------------------
    Xf = X[::stride, ::stride, ::stride].ravel()
    Yf = Y[::stride, ::stride, ::stride].ravel()
    Zf = Z[::stride, ::stride, ::stride].ravel()
    Uf = data[::stride, ::stride, ::stride].ravel()

    # ---------------------------
    # 컬러값 클리핑 (IQR)
    # ---------------------------
    ql, qh = np.quantile(Uf, clip_q)
    Uf_clip = np.clip(Uf, ql, qh)

    # ---------------------------
    # plot
    # ---------------------------
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        Xf, Yf, Zf,
        c=Uf_clip,
        cmap=cmap,
        s=6,
        alpha=0.9,
        linewidth=0
    )

    # ---------------------------
    # ⭐ 축 비율: 물리 길이 기준
    # ---------------------------
    ax.set_box_aspect([
        x_lin[-1] - x_lin[0],
        y_lin[-1] - y_lin[0],
        z_lin[-1] - z_lin[0],
    ])

    # ---------------------------
    # 라벨 / 타이틀
    # ---------------------------
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Displacement Field")

    # ---------------------------
    # 컬러바
    # ---------------------------
    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("Displacement")

    plt.tight_layout()
    plt.show()
