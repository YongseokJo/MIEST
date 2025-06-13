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







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimization():
    big_number = 1e5
    def __init__(self, sims, field, study_name, storage_name,
                 load_if_exists=True, monopole=True, level='zero',
                 field_aux=None, axis='sim', zdim=None, machine='cnn_enc_dec'):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field
        self.monopole       = monopole
        self.base_lr        = 1e-9 #base_lr
        self.field_aux      = field_aux
        self.axis           = axis
        self.zdim           = zdim
        self.machine        = machine

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
        else:
            print("Choose correct level!")
            raise

        self.opt_dir = 1
        if self.opt_dir == 2:
            self.directions     = ["minimize", "minimize"]
        else:
            self.directions     = ["minimize"]

        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = self.directions,
            sampler=sampler,
        )

    def objective(self,job_id,trial):
        device = torch.device("cuda:{}".format(job_id))
        hidden = trial.suggest_int("hidden", 1, 8)
        dr     = trial.suggest_float("dropout", 0.001, 0.5)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        max_lr = trial.suggest_float("lr", 1e-6, 5e-3, log=True)

        if self.zdim is None:
            z_dim = trial.suggest_int("z_dim", 50, 2000) ## 2000
            #z_dim = trial.suggest_int("z_dim", 50, 10000) ## 2000
        else:
            z_dim = self.zdim
        #z_dim      = 500
        beta       = self.beta
        gamma      = self.gamma
        # 1e-1,1e-2 for low
        # 1e+1,1e-0 for high
        # 1e+2,1e+1 for high2
        #1, 1e-1 for new
        channels = 1 if self.field_aux == None else 2

        mist = MIST(sim=self.sims, field=self.field, batch_size=200, 
                    data_type = 'image',
                    normalization=True,
                    monopole = self.monopole,
                    device=device, robust_axis=self.axis)
        success = mist.train(epochs=150,verbose=True, 
                             learning_rate=self.base_lr, max_lr=max_lr,
                             #which_machine="vib_cnn+cls", 
                             which_machine=self.machine, 
                             beta=beta, gamma=gamma,
                             channels=channels,
                             z_dim=z_dim, hidden1=hidden, dr=dr,
                             save_plot=False, save_model=False,
                            )
        num_trials = trial.number
        mist.save_models(fname="{}_{}".format(
            self.study_name, num_trials), fpath="./model/optuna")
        if success:
            if self.opt_dir == 2:
                valid_loss = mist.get_valid_loss()
                valid_auc  = mist.get_auc_score()
                return valid_loss, (valid_auc-0.5)**2
            else:
                valid_loss = mist.get_valid_loss()
                return valid_loss
        else:
            if self.opt_dir == 2:
                return self.big_number, self.big_number
            else:
                return self.big_number


    def run(self, n_trials, job_id, n_jobs):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = self.directions,
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
    # GPU prep
    n_gpu = print_gpu()

    # Arg Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-tng',        help='TNG on',
                        default=False, action='store_true')
    parser.add_argument('-simba',      help='SIMBA on',
                        default=False, action='store_true')
    parser.add_argument('-astrid',    help='ASTRID on',
                        default=False, action='store_true')
    parser.add_argument('-gadget',    help='ASTRID on',
                        default=False, action='store_true')
    parser.add_argument('-ramses',    help='ASTRID on',
                        default=False, action='store_true')
    parser.add_argument('-tng_sb',    help='TNG SB28 on',
                        default=False, action='store_true')
    parser.add_argument('-eagle',    help='EAGLE on',
                        default=False, action='store_true')
    parser.add_argument('--field',     type=str,  help='Field as input',
                        default='HI')
    parser.add_argument('--field_aux1', type=str,  help='Auxiliary field',
                        default=None)
    parser.add_argument('--field_aux2', type=str,  help='Auxiliary field',
                        default=None)
    parser.add_argument('--field_aux3', type=str,  help='Auxiliary field',
                        default=None)
    parser.add_argument('--level',     type=str,  help='Declassification level',
                        default='zero')
    parser.add_argument('--axis',     type=str,  help='Robust Axis',
                        default='sim')
    parser.add_argument('-monopole',   help='Include monopole?',
                        default=False, action='store_true')
    parser.add_argument('--machine',     type=str,  help='Which machine?',
                        default='cnn_enc_dec')
    parser.add_argument('--zdim',   help='set zdim',
                        default=None, type=int)
    args = parser.parse_args()

    sims = []
    if args.tng:
        sims += ['TNG']
    if args.simba:
        sims += ['SIMBA']
    if args.astrid:
        sims += ['ASTRID']
    if args.eagle:
        sims += ['EAGLE']
    if args.ramses:
        sims += ['RAMSES']
    if args.gadget:
        sims += ['GADGET']
    if args.tng_sb:
        sims += ['TNG_SB']
    if len(sims) == 1:
        sims = sims[0]

    fields = args.field
    if args.field_aux1 is not None:
        fields = [fields, args.field_aux1]
    if args.field_aux2 is not None:
        fields.append(args.field_aux2)
    if args.field_aux3 is not None:
        fields.append(args.field_aux3)

    #fields    = 'ne'
    #fields    = 'HI'
    #fields    = 'Mtot'
    #sims     = ['TNG', 'SIMBA']
    #sims     = ['TNG', 'ASTRID']


    n_trials=100; n_jobs=1;
    study_name  = get_study_name(sims,fields)
    print(study_name)
    study_name += "cnn_{}_monopole_{}_new_split".format(args.level,
                                                    int(args.monopole))
    if args.zdim is not None:
        study_name += f'_zdim_{args.zdim}'
    #study_name += "cnn_enc_dec_{}_monopole_{}_new_z_500".format(args.level,
    #                                                int(args.monopole))
    storage_name = "./database/{}.db".format(study_name)
    print(storage_name)
    opt = Optimization(sims, fields, study_name, storage_name,
                       monopole=args.monopole, level=args.level,
                       axis=args.axis, zdim=args.zdim, machine=args.machine)
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


