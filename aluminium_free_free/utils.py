import os

def isExistFolder(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def makeFolder(folder_path):
    if not isExistFolder(folder_path):
        os.makedirs(folder_path)
        return True
    return False

def minmax_scale(params,__min,__max):
    if __max - __min ==0:
        return params/__max/2
    scaled = (params - __min) / (__max - __min)
    return scaled

def inverse_minmax_scale(scaled,__min,__max):
    if __max - __min ==0:
        return params*__max*2
    params = scaled* (__max - __min) +__min
    return params


def feature_normalize(params,scale_info):

    Lx=scale_info['Lx']['max']
    Ly=scale_info['Ly']['max']
    Lz=scale_info['Lz']['max']


    params['Lx']=minmax_scale(params['Lx'],scale_info['Lx']['min'],scale_info['Lx']['max'])
    params['Ly']=minmax_scale(params['Ly'],scale_info['Ly']['min'],scale_info['Ly']['max'])
    params['Lz']= minmax_scale(params['Lz'],scale_info['Lz']['min'],scale_info['Lz']['max'])
    params['nx']=minmax_scale(params['nx'],scale_info['nx']['min'],scale_info['nx']['max'])
    params['ny']=minmax_scale(params['ny'],scale_info['ny']['min'],scale_info['ny']['max'])
    params['nz']=minmax_scale(params['nz'],scale_info['nz']['min'],scale_info['nz']['max'])
    params['xm0']=minmax_scale(params['xm0'],scale_info['xm0']['min'],scale_info['xm0']['max'])
    params['xm1']=minmax_scale(params['xm1'],scale_info['xm1']['min'],scale_info['xm1']['max'])
    params['ym0']=minmax_scale(params['ym0'],scale_info['ym0']['min'],scale_info['ym0']['max'])
    params['ym1']=minmax_scale(params['ym1'],scale_info['ym1']['min'],scale_info['ym1']['max'])
    params['zm0']=minmax_scale(params['zm0'],scale_info['zm0']['min'],scale_info['zm0']['max'])
    params['zm1']=minmax_scale(params['zm1'],scale_info['zm1']['min'],scale_info['zm1']['max'])
    params['E']=minmax_scale(params['E'],scale_info['E']['min'],scale_info['E']['max'])
    params['nu']=minmax_scale(params['nu'],scale_info['nu']['min'],scale_info['nu']['max'])
    params['rho']=minmax_scale(params['rho'],scale_info['rho']['min'],scale_info['rho']['max'])
    params['m_add']=minmax_scale(params['m_add'],scale_info['m_add']['min'],scale_info['m_add']['max'])
    params['freq']=minmax_scale(params['freq'],scale_info['freq']['min'],scale_info['freq']['max'])
    params['a_base']=minmax_scale(params['a_base'],scale_info['a_base']['min'],scale_info['a_base']['max']) 
    params['zeta']=minmax_scale(params['zeta'],scale_info['zeta']['min'],scale_info['zeta']['max'])


    return params,Lx, Ly, Lz


def standard_scale(params,__std,__mean,__max):
    if __std ==0:
        return params/__max/2
    scaled = (params - __mean) / (__std)
    return scaled

def inverse_standard_scale(scaled,__std,__mean,__max):
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

    sc = ax.scatter(
        x, y, z,
        c=u_vec,
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
