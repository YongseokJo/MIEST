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
dn=2
suffix='_l_{}_dn_{}'.format(L,dn)


field = "Xray"
norm = False
projection = False

fmaps = \
        "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/Maps_Xray_IllustrisTNG_LH_z=0.00.npy"
# read the data
maps = np.load(fmaps)
maps[maps<=0] = 1e-5
tng_maps = np.log10(maps)


if norm:
    maps = tng_maps
    nmaps = maps.reshape(maps.shape[0],-1)
    nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
    nmaps = nmaps.T.reshape(15000,256,256)
elif projection:
    nmaps = np.zeros((3000,256,256),dtype=float)
    ## projection axes
    for j in range(3):
        for i in range(5):
            nmaps[j::3,:,:] += tng_maps[i+5*j::15,:,:]
else:
    nmaps=tng_maps

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
    np.save("../data/wph_nIllustrisTNG_{}_for_vib_total{}".format(field,suffix),coef)
elif projection:
    np.save("../data/wph_pIllustrisTNG_{}_for_vib_total{}".format(field,suffix),coef)
else:
    np.save("../data/wph_IllustrisTNG_{}_for_vib_total{}".format(field,suffix),coef)

fmaps = \
        "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/Maps_Xray_SIMBA_LH_z=0.00.npy"
# read the data
maps = np.load(fmaps)
maps[maps<=0] = 1e-5
simba_maps = np.log10(maps)

if norm:
    maps = smba_maps
    nmaps = maps.reshape(maps.shape[0],-1)
    nmaps = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
    nmaps = nmaps.T.reshape(15000,256,256)
elif projection:
    nmaps = np.zeros((3000,256,256),dtype=float)
    ## projection axes
    for j in range(3):
        for i in range(5):
            nmaps[j::3,:,:] += simba_maps[i+5*j::15,:,:]
else:
    nmaps=simba_maps



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
    np.save("../data/wph_nSIMBA_{}_for_vib_total{}".format(field,suffix),coef)
elif projection:
    np.save("../data/wph_pSIMBA_{}_for_vib_total{}".format(field,suffix),coef)
else:
    np.save("../data/wph_SIMBA_{}_for_vib_total{}".format(field,suffix),coef)
