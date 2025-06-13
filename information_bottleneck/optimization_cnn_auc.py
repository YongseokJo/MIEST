import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *
import argparse
from joblib import Parallel, delayed, parallel_backend 
#from multiprocessing import Manager
#from dask.distributed import Client, wait
#from dask_cuda import LocalCUDACluster
from functools import partial





# Arg Parser
"""
parser = argparse.ArgumentParser()

parser.add_argument('--sims',    type=list,  help='Types of Uncertainties',
                   default='TNG')
parser.add_argument('--fields',   type=str,  help='Magnetitude of Uncertainties',
                    default='HI')
parser.add_argument('--L',     type=int, help='Do you want to restart?',
                   default=4)
parser.add_argument('--dn', type=int,  help='The number of epoch for restart',
                   default=2
                   )

args = parser.parse_args()
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimization():
    big_number = 1e5
    def __init__(self, sims, field, study_name, storage_name,
                 load_if_exists=True, monopole=True):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field
        self.monopole       = monopole
        self.base_lr        = 1e-9 #base_lr

        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize", "minimize"],
            #directions     = ["minimize"],
            sampler=sampler,
        )

    def objective(self,job_id,trial):
        device = torch.device("cuda:{}".format(job_id))
        hidden = trial.suggest_int("hidden", 3, 12)
        dr     = trial.suggest_float("dropout", 0.001, 0.5)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        max_lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)

        z_dim      = trial.suggest_int("z_dim", 50, 2000)
        beta       = trial.suggest_float("beta", 0.1,100)
        gamma      = trial.suggest_float("gamma", 1e-3,100)

        mist = MIST(sim=self.sims, field=self.field, batch_size=128,
                    data_type = 'image',
                    normalization=True,
                    monopole = self.monopole,
                    device=device)
        success = mist.train(epochs=150,verbose=True,
                             learning_rate=self.base_lr, max_lr=max_lr,
                             which_machine="vib_cnn+cls",
                             beta=beta, gamma=gamma,
                             z_dim=z_dim, hidden1=hidden, dr=dr,
                             save_plot=False, save_model=False,
                            )
        num_trials = trial.number
        mist.save_models(fname="{}_{}".format(
            self.study_name, num_trials), fpath="./model/optuna")
        if success:
            valid_loss = mist.get_valid_loss()
            valid_auc  = mist.get_auc_score()
            return valid_loss, (valid_auc-0.5)**2
        else:
            return self.big_number, self.big_number#\
                    #self.big_number, self.big_number, self.big_number


    def run(self, n_trials, job_id, n_jobs):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize", "minimize"],
            #directions     = ["minimize"],
            sampler=sampler,
        )
        study.optimize(partial(self.objective, job_id),
                       n_trials=n_trials, n_jobs=n_jobs)


def get_study_name(sims,field):
    study_name = ''

    if isinstance(sims, list):
        for sim in sims:
            study_name += sim + "_"
    else:
        study_name += sims + "_"

    if isinstance(fields, list):
        for field in fields:
            study_name += field + "_"
    else:
        study_name += fields + "_"

    return study_name

def print_gpu():
    n_gpu = torch.cuda.device_count()
    print('Num GPUs = ', n_gpu, file=sys.stderr)
    for i in range(n_gpu):
        print(torch.cuda.get_device_name(i), file=sys.stderr)
        gpu_properties = torch.cuda.get_device_properties(i)
        print(gpu_properties.name)
    return n_gpu

    """
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute\
          capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
           """


if __name__  == "__main__":
    n_gpu = print_gpu()
    sims     = 'TNG'
    fields    = 'HI'
    sims     = ['TNG', 'SIMBA']
    #sims     = ['TNG', 'ASTRID']
    n_trials=100; n_jobs=1;
    study_name = get_study_name(sims,fields)
    study_name += "cnn+cls_auc"
    storage_name = "./database/{}.db".format(study_name)
    opt = Optimization(sims, fields, study_name, storage_name,
                       monopole=True)
    print("Optimization Prepared!")

    r = Parallel(n_jobs=n_gpu)([delayed(opt.run)(n_trials, job_id=job_id, n_jobs=n_jobs)\
                                for job_id in range(n_gpu)])

    print('Number of finished trials: ', len(opt.study.trials))
    print('Best trial:')
    trial = opt.study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


