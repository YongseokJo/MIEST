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

fmaps = \
        "/mnt/home/fvillaescusa/CAMELS/Results/images_Astrid/Images_Mtot_Astrid_LH_z=0.00.npy"
# read the data
ast_maps = np.log10(np.load(fmaps))

maps = ast_maps
nmaps = maps.reshape(maps.shape[0],-1)
nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
#nmaps = (nmaps.T - nmaps.min(axis=1))/(nmaps.max(axis=1)-nmaps.min(axis=1)
nmaps = nmaps.T.reshape(15000,256,256)

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
N = int(15000/batch_size)
for i in range(N):
    wph = wph_op(nmaps[batch_size*i:batch_size*(i+1),:,:])
    try:
        coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()
    except:
        num_coeffs = wph.shape[1]
        coef = np.zeros((N*batch_size,num_coeffs), dtype=np.complexfloating)
        coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()
np.save("../data/wph_nAstrid_for_vib_total",coef)


for field in ['Mgas','Mstar','HI','ne','Vcdm','Z', 'T']:
#for field in ['T']:
    fmaps = \
            "/mnt/home/fvillaescusa/CAMELS/Results/images_Astrid/Images_{}_Astrid_LH_z=0.00.npy".format(field)
    # read the data
    maps = np.load(fmaps)
    maps[maps<=0] = 1e-5
    ast_maps = np.log10(maps)

    maps = ast_maps
    nmaps = maps.reshape(maps.shape[0],-1)
    nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
    #nmaps = (nmaps.T - nmaps.min(axis=1))/(nmaps.max(axis=1)-nmaps.min(axis=1)
    nmaps = nmaps.T.reshape(15000,256,256)

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
    N = int(15000/batch_size)
    for i in range(N):
        wph = wph_op(nmaps[batch_size*i:batch_size*(i+1),:,:])
        try:
            coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()
        except:
            num_coeffs = wph.shape[1]
            coef = np.zeros((N*batch_size,num_coeffs), dtype=np.complexfloating)
            coef[batch_size*i:batch_size*(i+1),:] = wph.cpu().detach().numpy()
    np.save("../data/wph_nAstrid_{}_for_vib_total".format(field),coef)