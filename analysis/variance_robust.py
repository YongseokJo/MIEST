import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import itertools

import torch
import gc
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
import argparse
warnings.filterwarnings('ignore')

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser()
parser.add_argument('-level', help='', default='zero', type=str)
args = parser.parse_args()




fpath = '/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck'
sims  = ['TNG', 'SIMBA']; field = 'HI'
#study_name = f"{sims[0]}_{sims[1]}_{field}_cnn_enc_dec_mid_monopole_1"
study_name = f"TNG_SIMBA_HI_cnn_{args.level}_monopole_1_new_split"

auc = []
storage    = f"sqlite:////mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
path       = f"/mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
if os.path.isfile(path):
    print(path, sims)
    mist = MIST(sim=sims, field=field, batch_size=32, data_type = 'image',
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
                b_om_0     = np.abs(np.mean((y_mean-y_true)/y_true)) ## bias
                std_om_0   = np.mean(y_std)
                y_true   = data[0][0][:,1]
                y_mean   = data[1][0][:,1]
                y_std    = data[2][0][:,1]
                y_res_sg_0 = np.mean(np.abs((y_mean-y_true)/y_true))*100
                b_sg_0     = np.abs(np.mean((y_mean-y_true)/y_true)) ## bias
                std_sg_0   = np.mean(y_std)

                print(y_res_om_0, y_res_sg_0)
                if y_res_om_0 > 6 or y_res_sg_0 > 6:
                    print("Skip!")
                    continue

                _, __auc, prob, _, _, _ = do_classification(mist)
                print(f"The AUC score is {__auc}.")
                print(prob)

                tmp = list()
                for sim_ext in ['ASTRID', 'EAGLE']:
                    data = mist.test_on_cnn(sim_ext, show_score=True, data_return=True,show_plot=False)
                    y_true = data[0][:,0]
                    y_mean = data[1][0][:,0]
                    y_std  = data[1][1][:,0]
                    y_res_om = np.mean(np.abs((y_mean-y_true)/y_true))*100
                    b_om     = np.abs(np.mean((y_mean-y_true)/y_true)) ## bias
                    std_om   = np.mean(y_std)
                    print(y_res_om)
    
                    y_true = data[0][:,1]
                    y_mean = data[1][0][:,1]
                    y_std  = data[1][1][:,1]
                    y_res_sg = np.mean(np.abs((y_mean-y_true)/y_true))*100
                    b_sg     = np.abs(np.mean((y_mean-y_true)/y_true)) ## bias
                    std_sg   = np.mean(y_std)
                    
                    tmp.extend([y_res_om, b_om, std_om, y_res_sg, b_sg, std_sg])
            
                auc.append([num_trial, __auc, 
                            y_res_om_0,
                            b_om_0, std_om_0, y_res_sg_0, b_sg_0,
                            std_sg_0,
                            tmp[0],
                            tmp[1], tmp[2], tmp[3],tmp[4],
                            tmp[5],
                            tmp[6], tmp[7],tmp[8], tmp[9], tmp[10],
                            tmp[11], prob[0], prob[1], prob[2],
                            prob[3]])
                print(auc[-1])
            except:
                continue
np.save(f"auc_{sims[0]}_{sims[1]}_{args.level}_3", auc)


