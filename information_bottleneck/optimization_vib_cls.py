import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimization():
    big_number = 1e5
    def __init__(self, sims, field, study_name, storage_name, load_if_exists=True):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field

    def objective(self,trial):
        z_dim      = trial.suggest_int("z_dim", 50, 2000)
        fe         = trial.suggest_float("fe", 0.01, 0.99)
        fd         = trial.suggest_float("fd", 0.01, 0.99)
        dr         = trial.suggest_float("dropout", 0.1, 0.9)
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        decay_rate = None
        beta       = trial.suggest_float("beta", 1,1000,log=True)
        gamma      = trial.suggest_float("gamma", 1e-3,1e-1,log=True)

        mist = MIST(sim=self.sims, field=self.field, batch_size=128, 
                    normalization=True,
                    monopole = True,
                    L=4, dn=0,
                    projection=True,
                    device=device)
        success = mist.train(epochs=3000,
                             verbose=True, learning_rate=lr, decay_rate=decay_rate,
                             which_machine="vib+cls", 
                             beta=beta, gamma=gamma,
                             z_dim=z_dim, hidden1=fe, hidden2=fd, dr=dr,
                             save_plot=False, save_model=False,
                            )
        num_trials = trial.number
        mist.save_models(fname="{}_{}".format(
            self.field, num_trials), fpath="./model/optuna")
        if success:
            valid_loss = mist.get_valid_loss()
            auc_socre  = mist.get_auc_score()
            return valid_loss, auc_socre # , chi2_om, chi2_sig, auc
        else:
            return self.big_number#\
                    #self.big_number, self.big_number, self.big_number


    def run(self):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler(n_startup_trials=20)
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize"]*2,
            sampler=sampler,
        )
        study.optimize(self.objective, n_trials=1000, n_jobs=10)
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("Best value: {} (params: {})\n".format(study.best_value,
                                                     study.best_params))


if __name__  == "__main__":
    sims         = ['TNG', 'SIMBA']
    #sims         = ['SIMBA', 'ASTRID']
    #sims         = ['TNG', 'ASTRID']
    field        = 'HI'
    storage_name = "opt_VIB_CLS.db"
    study_name   = ""
    for sim in sims:
        study_name += sim + "_"
    study_name  += field

    opt = Optimization(sims, field, study_name, storage_name)
    print("Optimization Prepared!")
    opt.run()
