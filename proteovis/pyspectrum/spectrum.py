"""スペクトル解析用ライブラリ

ベースライン補正やノイズ除去を目的とする

Todo:
"""

import numpy as np
from scipy import signal
from scipy.sparse import csc_matrix, spdiags
import scipy.sparse.linalg as spla
import scipy.interpolate as scipl

#パラメタを入力します、うまく推定ができないときはここをいじってください
#AsLSでのベースライン推定は ( W(p) + lam*D'D )z = Wy のとき、重み p と罰則項の係数 lam がパラメタです
#Savitzky-Golyでは、測定値をいくつに分割するかを dn で設定し（窓の数は len(Y)/dn になります)、多項式次数を poly で設定します
# paramAsLS = [ lam , p ]
# paramSG   = [ dn , poly ]
paramAsLS = [10**2, 0.001]
paramSG = [40, 5]

class CorrectSpec:
    
    """スペクトルをきれいにする
    
    現状はベースライン補正とノイズ除去のみ
    
    Attributes:
    """
    pass

    def __init__(self,lam=10**2,p=0.001,dn=40,poly=5):
        """
        lamとpはベースライン補正のパラメータ
        dnとpolyはノイズ除去のパラメータ

        Args:
            lam (float): 罰則項の係数 10^2 - 10^9
            p (float): 重み 0.001 - 0.1
            dn (int): スペクトル長さ/dn > poly
            poly (int): 補正係数
        """
        self.lam = lam
        self.p = p
        self.dn = dn
        self.poly = poly
    
    def remove_baseline(self,y):
        bkg = baseline_als(y,self.lam, self.p)
        fix = y - bkg
        return fix
    
    def remove_noise(self,y):
        smth = SGs(y, self.dn, self.poly)
        return smth
    
    def clean_spec(self,y):
        #fix = self.remove_baseline(y)
        smth = self.remove_noise(y)
        fix = self.remove_baseline(smth)
        return fix



#AsLSによりベースライン推定を行います
def baseline_als(y, lam, p, niter=10):
    #https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    #p: 0.001 - 0.1, lam: 10^2 - 10^9
    # Baseline correction with asymmetric least squares smoothing, P. Eilers, 2005
    L = len(y)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spla.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

#Savitzky-Golyによりノイズ除去を行います
def SGs(y,dn,poly):
    # y as np.array, dn as int, poly as int
    n = len(y) // dn
    if n % 2 == 0:
        N = n+1
    elif n % 2 == 1:
        N = n
    else:
        print("window length can't set as odd")
    SGsmoothed = signal.savgol_filter(y, window_length=N, polyorder=poly)
    return SGsmoothed