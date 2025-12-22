import os

def isExistFolder(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def makeFolder(folder_path):
    if not isExistFolder(folder_path):
        os.makedirs(folder_path)
        return True
    return False
def feature_normalize(params):

    Lx=20
    Ly=20
    Lz=20


    params['Lx']=params['Lx']/Lx
    params['Ly']=params['Ly']/Ly
    params['Lz']=params['Lz']/Lz
    params['nx']=params['nx']/100
    params['ny']=params['ny']/100
    params['nz']=params['nz']/100
    params['xm0']=params['xm0']/20
    params['xm1']=params['xm1']/20
    params['ym0']=params['ym0']/20
    params['ym1']=params['ym1']/20
    params['zm0']=params['zm0']/20
    params['zm1']=params['zm1']/20
    params['E']=params['E']/300e9
    params['nu']=params['nu']
    params['rho']=params['rho']/10000
    params['m_add']=params['m_add']/1000
    params['freq']=params['freq']/4096
    params['a_base']=params['a_base']/1000
    params['zeta']=params['zeta']

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
