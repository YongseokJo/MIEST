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


import warnings
warnings.filterwarnings('ignore')

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import gc
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-num', help='', default=0, type=int)
args = parser.parse_args()
comb_num = args.num



sims = ['TNG', 'SIMBA', 'ASTRID', 'EAGLE']; field='HI'
study_names = ["HI_cnn_enc_dec_zero_monopole_1","HI_cnn_zero_monopole_1_new","HI_cnn_zero_monopole_1_new_split"]
fpath = '/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck'


combinations = list(itertools.combinations(sims, 2))
externals = [[x for x in sims if x not in comb] for comb in combinations]
data   = list()
params = list()
label  = list()

comb = combinations[comb_num]
print(comb)

auc = []
for fname in study_names:
    study_name = f"{comb[0]}_{comb[1]}_" + fname
    storage    = f"sqlite:////mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
    path       = f"/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
    if os.path.isfile(path):
        print(path, comb)
        mist = MIST(sim=[comb[0],comb[1]], field=field, batch_size=32, data_type = 'image',
                    normalization=True, monopole = True, device=device)
        for num_trial in range(1000):
            torch.cuda.empty_cache()
            gc.collect()
            path_trial = f"/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/model/optuna/{study_name}_{num_trial}_vib.pt"
            if os.path.isfile(path_trial): 
                print("file exists!")
                try:
                    mist.load_optuna_models(storage=storage,study_name=study_name, which_machine="cnn_enc_dec",num_trial=num_trial, fpath=fpath)
                    data = mist.make_plots_cnn(fname='cnn_test', save_plot=False, data_return=True,show_plot=False)

                    y_true   = data[0][0][:,0]
                    y_mean   = data[1][0][:,0]
                    y_std    = data[2][0][:,0]
                    y_res_om_0 = np.mean(np.abs((y_mean-y_true)/y_true))*100
                    y_true   = data[0][0][:,1]
                    y_mean   = data[1][0][:,1]
                    y_std    = data[2][0][:,1]
                    y_res_sg_0 = np.mean(np.abs((y_mean-y_true)/y_true))*100

                    if np.isnan(y_res_om_0) or np.isnan(y_res_sg_0):
                        print(y_res_om_0, y_res_sg_0)
                        continue

                    if y_res_om_0 > 6 or y_res_sg_0 > 6:
                        print(y_res_om_0, y_res_sg_0)
                        continue

                    _, __auc, prob, _, _, _ = do_classification(mist)
                    print(prob)
                    print(f"The AUC score is {__auc}.")

                    auc.append([num_trial, y_res_om_0, y_res_sg_0, __auc,
                                prob[0], prob[1], prob[2], prob[3]])
                except:
                    raise
np.save(f"auc_{comb[0]}_{comb[1]}", auc)

