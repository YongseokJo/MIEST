import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import sklearn
import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from imp import reload 
# Change in mymodule/'
import vib_utils
reload(vib_utils)
from vib_utils import *
import mist_utils
reload(mist_utils)
from mist_utils import *
import umap_utils
reload(umap_utils)
from umap_utils import *

import warnings
warnings.filterwarnings('ignore')

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/myutils/"))
from plt_utils import generateAxesForMultiplePlots, remove_inner_axes
# Dimension reduction and clustering libraries
import umap
import umap.plot
#import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from plt_utils import generateAxesForMultiplePlots
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['font.family']='serif'




#sim  = ['TNG', 'SIMBA', 'ASTRID']; field = 'HI'
#study_name = "TNG_SIMBA_ASTRID_HI_cnn_mid_monopole_1_new"

sim  = ['TNG', 'SIMBA']; field = 'HI'
study_name = "TNG_SIMBA_HI_cnn_enc_dec_mid_monopole_1"
fpath = '/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck'
vib_umap = UMAP(sim=sim,field=field,study_name=study_name, fpath=fpath,
                ext_sim='ASTRID')

ext_sim = False
mapper, data, pred, label = vib_umap.return_umap(n_neighbors=100,min_dist=1,
                                                 n_components=24,
                                                 random_state=10,
                                                 ext_sim=ext_sim,
                                                 which_data='train')

colors  = ['b','r','g']
markers = ['o','s','*']
params  = [r"$\Omega_m$", r"$\sigma_8$"]
x=pred[:,0]; y=pred[:,1]; z=pred[:,2]

alpha = 1; s=3;
dpi   = 200;
N     = 200;

theta = np.append(np.linspace(-180,180,int(N/2)),[180]*int(N/2))
phi   = np.append([-180]*int(N/2),np.linspace(-180,180,int(N/2)))

for i in range(N):
    fig = plt.figure(figsize=(30,10))
    fig.tight_layout(pad=-10)
    for j in range(3):
        ax = fig.add_subplot(131+j, projection='3d')

        if j == 0:
            ax.scatter(x, y, z, c=label[0], cmap='Spectral', marker='o', s=s, alpha=alpha)

            ax.view_init(theta[i], phi[i])
            ax.axis('off')

        if j > 0:
            c=label[1][:,j-1]
            for k in range(len(sim)+int(ext_sim)):
                index = label[0] == colors[k]
                cbar = ax.scatter(x[index], y[index], z[index], c=c[index],
                           cmap='Spectral', marker=markers[k],s=s,alpha=alpha)

                axins_l = inset_axes(ax,
                                     width="100%",  # width = 5% of parent_bbox width
                                     height="3%",  # height : 50%
                                     loc='lower left',
                                     bbox_to_anchor=(.2, .15, .6, 1),
                                     bbox_transform=ax.transAxes,
                                     borderpad=0.0,
                                    )
                cbar_dens=plt.colorbar(cbar, cax=axins_l, orientation='horizontal', aspect='auto')
                cbar_dens.ax.yaxis.set_ticks_position('right')
                cbar_dens.ax.tick_params(axis='y',labelsize=20)
                cbar_dens.set_label(params[j-1], labelpad=10, fontsize=20)

                ax.view_init(theta[i], phi[i])
                ax.axis('off')
    #plt.savefig(f'movie_img/TNG_SIMBA_ASTRID_mid_param_{i}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'movie_img/TNG_SIMBA_mid_param_{i}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


