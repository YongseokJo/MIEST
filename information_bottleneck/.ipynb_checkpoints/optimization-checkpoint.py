import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import optuna
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimization():
    def __init__(self, sims, field, study_name, storage_name, load_if_exists=True):
        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.sims           = sims
        self.field          = field

    def objective(self,trial):
        mist = MIST(sim=self.sims, field=self.field, device=device)
        trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        beta = trial.suggest_float("beta", 1e-2,1000)
        gamma = trial.suggest_float("gamma", 1e-2,100)

        mist.train(epochs=5000,
                   verbose=True, learning_rate=1e-3, decay_rate=1.0,
                   which_machine="vib+cls", 
                   beta=beta,
                   gamma=gamma, 
                   save_plot=False, save_model=False,
                  )
        mse, r2, auc = mist.get_score(bias_test=False, _print=False)
        return mse, r2, auc

    def run(self):
        print("Optimization Starts!")
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = ["minimize", "maximize", "minimize"])
        study.optimize(self.objective, n_trials=100)
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("Best value: {} (params: {})\n".format(study.best_value,
                                                     study.best_params))


if __name__  == "__main__":
    sims         = ['TNG', 'SIMBA']
    field        = 'HI'
    storage_name = "loss_opt.db"
    study_name   = ""
    for sim in sims:
        study_name += sim + "_"
    study_name  += field

    opt = Optimization(sims, field, study_name, storage_name)
    print("Optimization Prepared!")
    opt.run()
