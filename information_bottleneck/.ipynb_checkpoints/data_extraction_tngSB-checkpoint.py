import pywph as pw
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
import sklearn
import matplotlib.pyplot as plt

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for field in ['Mgas','Mstar','HI','ne','Vcdm','Z', 'T']:
    index= []
    try:
        del sb_maps
    except:
        pass
    
    for i in range(1000):
        fname =\
        "/mnt/home/fvillaescusa/CAMELS/Results/images_IllustrisTNG_SB28/Images_{}_IllustrisTNG_SB28_SB28_{}_z=0.00.npy"\
        .format(field,i)
        try:
            sb_maps = np.r_[sb_maps, np.log10(np.load(fname))]
            index.append(i)
        except NameError:
            sb_maps = np.log10(np.load(fname))
            index.append(i)
        except FileNotFoundError:
            continue

    maps = sb_maps
    nmaps = maps.reshape(maps.shape[0],-1)
    nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
    #nmaps = (nmaps.T - nmaps.min(axis=1))/(nmaps.max(axis=1)-nmaps.min(axis=1)
    nmaps = nmaps.T.reshape(15*len(index),256,256)

    M, N = 256, 256
    J = 7
    L = 4
    dn = 2
    try: 
        del coef
    except:
        pass
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0)
    batch_size = 100
    N = int(15*len(index)/batch_size)
    for i in range(N):
        wph = wph_op(nmaps[batch_size*i:min(15*len(index),batch_size*(i+1)),:,:])
        try:
            coef[batch_size*i:min(15*len(index),batch_size*(i+1)),:] = wph.cpu().detach().numpy()
        except:
            num_coeffs = wph.shape[1]
            coef = np.zeros((15*len(index),num_coeffs), dtype=np.complexfloating)
            coef[batch_size*i:min(15*len(index),batch_size*(i+1)),:] = wph.cpu().detach().numpy()
    np.save("../data/wph_nTNG_SB28_{}_for_vib_total".format(field),coef)
