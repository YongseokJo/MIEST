import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *
import argparse
from joblib import Parallel, delayed, parallel_backend 
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
                 load_if_exists=True, L=4, dn=2, monopole=True,
                 projection=False, level='zero'):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field
        self.L              = L
        self.dn             = dn
        self.monopole       = monopole
        self.projection     = projection

        if level == 'zero':
            self.beta, self.gamma = 0, 0
        elif level == 'low':
            self.beta, self.gamma = 1e-1, 1e-2
        elif level == 'mid':
            self.beta, self.gamma = 1e+0, 1e-1
        elif level == 'high':
            self.beta, self.gamma = 1e+1, 1e+0
        elif level == 'high2':
            self.beta, self.gamma = 1e+2, 1e+1
        elif level == 'high3':
            self.beta, self.gamma = 1e+3, 1e+2
        elif level == 'extrm':
            self.beta, self.gamma = 1e+6, 1e+5
        else:
            print("Choose corrent level!")
            raise

        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize"],
            sampler=sampler,
        )

    def objective(self,job_id,trial):
        device = torch.device("cuda:{}".format(job_id))
        z_dim      = trial.suggest_int("z_dim", 50, 2000)
        fe         = trial.suggest_float("fe", 0.01, 1.00)
        fd         = trial.suggest_float("fd", 0.01, 1.00)
        dr         = trial.suggest_float("dropout", 0.001, 0.01)
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        #lr         = 0.002

        decay_rate = None
        #beta       = 0 #trial.suggest_float("beta", 1,1000)
        #gamma      = 1e-2 #trial.suggest_float("gamma", 1e-2,100)
        #jgamma      = trial.suggest_float("gamma", 1e-4,1e-1,log=True)
        average    = True

        mist = MIST(sim=self.sims, field=self.field, batch_size=100,
                    normalization=True,
                    monopole = self.monopole,
                    L=self.L, dn=self.dn,
                    projection=self.projection,
                    average=average,
                    proc_imag=np.absolute,
                    data_type='wph',
                    device=device)
        success = mist.train(epochs=7000,
                             verbose=True, learning_rate=lr, #decay_rate=decay_rate,
                             which_machine="vib+cls", 
                             #which_machine="vib", 
                             beta=self.beta, gamma=self.gamma,
                             z_dim=z_dim, hidden1=fe, hidden2=fd, dr=dr,
                             save_plot=False, save_model=False,
                            )
        num_trials = trial.number
        mist.save_models(fname="{}_{}".format(
            self.study_name, num_trials), fpath="./model/optuna")
        if success:
            valid_loss = mist.get_valid_loss()
            return valid_loss # , chi2_om, chi2_sig, auc
        else:
            return self.big_number#\
                    #self.big_number, self.big_number, self.big_number



    def run(self, n_trials, job_id, n_jobs):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            #directions     = ["minimize", "minimize"],
            directions     = ["minimize"],
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



if __name__  == "__main__":
    # GPU prep
    n_gpu = print_gpu()

    # Arg Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-tng',        help='TNG on',
                        default=False, action='store_true')
    parser.add_argument('-simba',      help='SIMBA on',
                        default=False, action='store_true')
    parser.add_argument('--astrid',    help='ASTRID on',
                        default=False, action='store_true')
    parser.add_argument('--field',     type=str,  help='Field as input',
                        default='HI')
    parser.add_argument('--field_aux', type=str,  help='Auxiliary field',
                        default=None)
    parser.add_argument('--level',     type=str,  help='Declassification level',
                        default='zero')
    parser.add_argument('-monopole',   help='Include monopole?',
                        default=False, action='store_true')
    args = parser.parse_args()

    sims = []
    if args.tng:
        sims += ['TNG']
    if args.simba:
        sims += ['SIMBA']
    if args.astrid:
        sims += ['ASTRID']
    if len(sims) == 1:
        sims = sims[0]

    if args.field_aux is not None:
        fields = [args.field, args.field_aux]
    else:
        fields = args.field

    L=4; dn=0; n_trials=500; n_jobs=6;
    projection=False;
    study_name = get_study_name(sims,fields)
    study_name +=\
            "wph_{}_new_mu_dn_0".format(args.level)
    storage_name = "./database/{}.db".format(study_name)
    print(study_name)
    print(storage_name)
    opt = Optimization(sims, fields, study_name, storage_name,
                       L=L, dn=dn, level=args.level, monopole=args.monopole, projection=projection)
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

