import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import itertools

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

import argparse
import warnings
warnings.filterwarnings('ignore')

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--sim', type=str,  help='Simulation',
                    default='TNG')
args = parser.parse_args()



sims = ['TNG', 'SIMBA', 'ASTRID', 'EAGLE']; field='HI'
fpath = '/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck'

externals = [[tmp2 for tmp2 in sims if tmp is not tmp2] for tmp in sims]
data = []
ext  = []
#for i, sim in enumerate(sims):




thres = 0
i = 3

sim = args.sim
if sim == 'TNG':
    i = 0
if sim == 'SIMBA':
    i = 1
if sim == 'ASTRID':
    i = 2
if sim == 'EAGLE':
    i = 3


rel_err     = []
crs_rel_err = []
study_name = f"{sim}_HI_cnn_zero_monopole_1_new_split"
storage    = f"sqlite:////mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
path       = f"/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
for num_trial in range(1000):
    path_trial = f"/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/model/optuna/{sim}_HI_cnn_zero_monopole_1_new_split_{num_trial}_vib.pt"
    if os.path.isfile(path) and os.path.isfile(path_trial):
        print("file exists!")
        mist = MIST(sim=[sim], field=field, batch_size=32, 
                    data_type = 'image',
                    normalization=True,
                    monopole = True,
                    device=device, robust_axis='sim')
        try:
            mist.load_optuna_models(storage=storage,study_name=study_name, which_machine="vib_cnn",num_trial=num_trial, fpath=fpath)
            #mse_om, mse_sig, _,_,_ = mist.get_score_cnn()
            #if mist.trial.values[0] > thres:
            #    print("values =", mist.trial.values[0])
            #    continue
            data = mist.make_plots_cnn(fname='cnn_test', save_plot=False, data_return=True,show_plot=False)
            #print(data)
            y_true   = data[0][0][:,0]
            y_mean   = data[1][0][:,0]
            y_std    = data[2][0][:,0]
            y_res_om_0 = np.mean(np.abs((y_mean-y_true)/y_true))*100
            y_true   = data[0][0][:,1]
            y_mean   = data[1][0][:,1]
            y_std    = data[2][0][:,1]
            y_res_sg_0 = np.mean(np.abs((y_mean-y_true)/y_true))*100

            print(y_res_om_0)
            if y_res_om_0 > 6 or y_res_sg_0 > 6:
                #thres = min(mist.trial.values[0], thres)
                #print(y_res_om_0, y_res_sg_0, thres)
                #continue
                pass

            tmp = list()
            for sim_ext in externals[i]:
                data = mist.test_on_cnn(sim_ext, show_score=True, data_return=True,show_plot=False)
                y_true = data[0][:,0]
                y_mean = data[1][0][:,0]
                y_std  = data[1][1][:,0]
                y_res_om = np.mean(np.abs((y_mean-y_true)/y_true))*100
                print(y_res_om)

                y_true = data[0][:,1]
                y_mean = data[1][0][:,1]
                y_std  = data[1][1][:,1]
                y_res_sg = np.mean(np.abs((y_mean-y_true)/y_true))*100

                tmp.append([num_trial, y_res_om, y_res_sg])
            crs_rel_err.append(tmp)
            rel_err.append([num_trial, y_res_om_0, y_res_sg_0])
            print(rel_err[-1], crs_rel_err[-1])
        except:
            continue
np.save(f"rel_err_{sim}_all", rel_err)
np.save(f"crs_rel_err_{sim}_all", crs_rel_err)



