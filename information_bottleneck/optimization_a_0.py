import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *
import argparse

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
        dr      = trial.suggest_float("dropout", 0.0, 0.9)
        hidden  = trial.suggest_float("hidden", 0.5,2)
        z_dim   = trial.suggest_int("z_dim", 50, 2000)

        lr         = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        beta       = 0
        gamma      = 0
        #decay_rate = trial.suggest_float("decay", 0.8, 0.97)
        decay_rate = 0.97

        mist = MIST(sim=self.sims, field=self.field, batch_size=100,
                   normalization=True,
                   monopole = True,
                    device=device)
        success = mist.train(epochs=3000,
                             verbose=True, learning_rate=lr, decay_rate=decay_rate,
                             which_machine="vib+cls_a", 
                             beta=beta, gamma=gamma,
                             z_dim=z_dim, hidden=hidden, dr=dr,
                             save_plot=False, save_model=False,
                            )
        num_trials = trial.number
        mist.save_models(fname="{}_{}_a_0".format(field, num_trials), fpath="./model/optuna")
        if success:
            mse_om, mse_sig, chi2_om, chi2_sig, auc = mist.get_score(bias_test=False, _print=False)
            return mse_om, mse_sig,auc#, chi2_om, chi2_sig, auc
        else:
            return self.big_number, self.big_number, self.big_number#,\

    def run(self):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler(n_startup_trials=20)
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize"]*3,
            sampler=sampler,
        )
        study.optimize(self.objective, n_trials=500, n_jobs=10)
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("Best value: {} (params: {})\n".format(study.best_value,
                                                     study.best_params))


if __name__  == "__main__":
    # Arg Parser
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--field',    type=str,  help='Field of interest')
    #args = parser.parse_args()



    sims         = ['TNG', 'SIMBA']
    field        =  'T'
    storage_name = "opt_VIB_a_0_m.db"
    study_name   = ""
    for sim in sims:
        study_name += sim + "_"
    study_name  += field

    opt = Optimization(sims, field, study_name, storage_name)
    print("Optimization Prepared!")
    opt.run()
