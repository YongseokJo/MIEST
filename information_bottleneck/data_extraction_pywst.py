import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils

from kymatio.torch import Scattering2D
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D
from kymatio.scattering3d.utils \
    import generate_weighted_sum_of_gaussians
from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir


# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu' # temporarily
# Fix random seeds for reproducibility
seed = 73
torch.manual_seed(seed)
np.random.seed(seed)



def scattering_transform(grid_2d, device,J=1,L=1,max_order=2,sigma=1,
                         integral_powers=[2],shape=(128,128,128),batch_size=100):

    scattering = Scattering2D(J=J, shape=shape, L=L, max_order=max_order)
    scattering.to(device)

    n_size = grid_2d.shape[0]
    n_batches = int(np.ceil(n_size / batch_size))

    this_time = None
    last_time = None
    coefficients = None
    for i in range(n_batches):
        # Extract the current batch.
        start = i * batch_size
        end = min(start + batch_size, n_size)

        full_density_batch = grid_2d[start:end,:,:]
        full_density_batch = torch.from_numpy(full_density_batch)
        full_density_batch = full_density_batch.to(device).float()

        # Compute scattering coefficients 
        full_scattering = scattering(full_density_batch)
        print(full_scattering.shape)
        if coefficients is None:
            coefficients =\
                    full_scattering.cpu().detach().numpy()
        else:
            coefficients =\
                    np.concatenate([coefficients,
                                    full_scattering.cpu().detach().numpy()],
                                   axis=0)
    return coefficients

#---------------------------------------------------------------
# Main
#---------------------------------------------------------------

# Sim set
fields = ['Mtot', 'T', 'Mgas','Mstar','HI','ne','Vcdm','Z']
simulations = ['TNG', 'SIMBA', 'ASTRID']
sim_names = {'TNG':'IllustrisTNG',
             'SIMBA':'SIMBA',
             'ASTRID':'Astrid'}
fmaps = "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps"

# Scatter transform set
max_order=2;
J=7;L=5;shape = (256,256)
batch_size = 100

raw  = False
log  = True
norm = False
prefix = ''



for sim in simulations:
    for field in fields:
        # read the data
        fmap = fmaps+f'_{field}_{sim_names[sim]}_LH_z=0.00.npy'
        grid_2d = np.log10(np.load(fmap))

        # Generating coefficients
        coef  = scattering_transform(grid_2d, device, J=J, L=L, max_order=2,
                                     sigma=1,integral_powers=[2],shape=shape,batch_size=batch_size)
        np.save("../data/pywst_{}{}_{}_for_vib_total_".format(prefix,sim_names[sim],field),coef)
        print("Coefficients saved.")



