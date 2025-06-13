import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *
import argparse
from joblib import Parallel, delayed


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
                 load_if_exists=True, L=4, dn=0, monopole=True, projection=True):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field
        self.L              = L
        self.dn             = dn
        self.monopole       = monopole
        self.projection     = projection

        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize"],
            sampler=sampler,
        )

    def objective(self,trial):
        z_dim      = trial.suggest_int("z_dim", 50, 2000)
        fe         = trial.suggest_float("fe", 0.01, 1.00)
        fd         = trial.suggest_float("fd", 0.01, 1.00)
        dr         = trial.suggest_float("dropout", 0.001, 0.9)
        lr         = trial.suggest_float("lr", 1e-8, 1e-2, log=True)

        decay_rate = None
        beta       = 0 #trial.suggest_float("beta", 1,1000)
        #gamma      = 1e-2 #trial.suggest_float("gamma", 1e-2,100)
        gamma      = trial.suggest_float("gamma", 1e-4,1e-1,log=True)
        average    = False

        mist = MIST(sim=self.sims, field=self.field, batch_size=32, 
                    normalization=True,
                    monopole = self.monopole,
                    L=self.L, dn=self.dn,
                    projection=self.projection,
                    average=average,
                    proc_imag=np.absolute,
                    device=device)
        success = mist.train(epochs=2000,
                             verbose=True, learning_rate=lr, decay_rate=decay_rate,
                             #which_machine="vib+cls", 
                             which_machine="vib", 
                             beta=beta, gamma=gamma,
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


    def run(self, n_trials):
        print("Optimization Starts!")
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=1)
        #print("Study statistics: ")
        #print("  Number of finished trials: ", len(study.trials))
        #print("Best value: {} (params: {})\n".format(study.best_value,
        #                                             study.best_params))

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


if __name__  == "__main__":
    sims     = 'TNG'
    sims     = 'EAGLE'
    #sims     = ['TNG', 'SIMBA']
    fields   = 'HI'; L=10; dn=0; n_trials=20;
    monopole = True; projection=True;
    study_name = get_study_name(sims,fields)
    study_name +=\
            "l_{}_dn_{}_m_{}_p_{}".format(L,dn,int(monopole),int(projection))
    storage_name = "./database/{}.db".format(study_name)
    opt = Optimization(sims, fields, study_name, storage_name,
                       L=L, dn=dn, monopole=monopole, projection=projection)
    print("Optimization Prepared!")

    r = Parallel(n_jobs=20)([delayed(opt.run)(n_trials) for _ in range(20)])

    print('Number of finished trials: ', len(opt.study.trials))
    print('Best trial:')
    trial = opt.study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

