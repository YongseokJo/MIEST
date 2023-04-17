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

X, Y = 256, 256
J = 7
L = 4 ##
dn= 0
suffix='_l_{}_dn_{}'.format(L,dn)

norm = False
projection = False

for field in ['Mtot', 'Mgas','Mstar','HI','ne','Vcdm','Z', 'T']:
    fmaps = \
            "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_Astrid_LH_z=0.00.npy".format(field)
    # read the data
    maps = np.load(fmaps)
    maps[maps<=0] = 1e-5
    ast_maps = np.log10(maps)

    maps = ast_maps
    if norm:
        nmaps = maps.reshape(maps.shape[0],-1)
        nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
        #nmaps = (nmaps.T - nmaps.min(axis=1))/(nmaps.max(axis=1)-nmaps.min(axis=1)
        nmaps = nmaps.T.reshape(15000,256,256)
    elif projection:
        nmaps = np.zeros((3000,256,256),dtype=float)
        for j in range(3): ## projection axes
            for i in range(5):
                nmaps[j::3,:,:] += ast_maps[i+5*j::15,:,:]
    else:
        nmaps = ast_maps

    try: 
        del coef
    except:
        pass
    wph_op = pw.WPHOp(X, Y, J, L=L, dn=dn, device=0)
    batch_size = 100
    if projection:
        N = int(3000/batch_size)
    else:
        N = int(15000/batch_size)
    for i in range(N):
        wph = wph_op(nmaps[batch_size*i:batch_size*(i+1),:,:])
        try:
            coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()
        except:
            num_coeffs = wph.shape[1]
            coef = np.zeros((N*batch_size,num_coeffs), dtype=np.complexfloating)
            coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()

    if norm:
        np.save("../data/wph_nAstrid_{}_for_vib_total{}".format(field,suffix),coef)
    elif projection:
        np.save("../data/wph_pAstrid_{}_for_vib_total{}".format(field,suffix),coef)
    else:
        np.save("../data/wph_Astrid_{}_for_vib_total{}".format(field,suffix),coef)

