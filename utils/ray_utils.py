import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
import sklearn
from torch.utils.data.sampler import SubsetRandomSampler
import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from callback import *
from networks import *
from copy import deepcopy

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu' # temporarily
# Fix random seeds for reproducibility


###################
### Loss functions
###################
def no_vib_loss(y_mean,y_sigma, y):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.sum((torch.sum(J1,axis=0)))
    return J0+J1 / y.size(0), J0, J1


def vib_loss(y_mean,y_sigma, y, mu, std, gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.mean(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.mean((torch.sum(J1,axis=0)))
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return ((gamma*KL + J0+J1) / y.size(0), J0, J1, KL)


def vib_cls_loss(y_mean,y_sigma, y, y_class, _class, mu, std, beta=0.01,
                 gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.mean(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.mean((torch.sum(J1,axis=0)))
    CrEn = nn.CrossEntropyLoss()
    #NL   = nn.NLLLoss()
    CE   = CrEn(y_class, _class)
    #CE   = NL(y_class, _class)
    KL   = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)/y.size(0)
    return (J0+J1-beta*CE+gamma*KL, J0, J1, -beta*CE, -gamma*KL)


def adaptive_vib_loss(y_mean, y_sigma, y, mu, std, gamma=0.01):
    J0   = torch.mean((y-y_mean)**2).mean()
    J1   = torch.mean(((y-y_mean)**2-y_sigma**2)**2).mean()
    KL   = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)/y.size(0)
    return (J0+J1+gamma*KL, J0, J1, gamma*KL)






class RayTrainer:
    def __init__(self, params):
        # Some infos
        self.machine = {'vib':VIB, 'vib+cls':VIB}
        self.is_cls  = ['vib+cls']
        self.loss    = {}
        self.names_loss = ['total_loss', 'j0_loss', 'j1_loss', 'ce0_loss',
                      'ce1_loss']

        # Arguments
        self.which_machine = params["which_machine"]
        self.train_loader  = params["trainset"]
        self.val_dataset   = params["validset"]
        self.num_sim       = params["num_sim"]
        self.max_epoch     = params["max_epoch"]
        self.input_shape   = params["input_shape"]
        self.output_shape  = params["output_shape"]
        self.patience      = params["patience"]

        self.num_loss      = 5 if self.which_machine in self.is_cls else 4

        # Early Stopper
        self.log_container      = Log([],batch_size=self.train_loader.batch_size)
        self.early_stopping     = EarlyStopping(None, tol=None, patience=self.patience)
        self.callback_container = CallbackContainer([self.log_container,
                                                     self.early_stopping])
        self.callback_container.set_trainer(self)


    def run(self, config, checkpoint_dir=None):
        # Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_epoch = 0
        save_plot = False

        # Network
        self.vib = self.machine[self.which_machine](input_shape=self.input_shape,
                                     output_shape=self.output_shape,
                                     z_dim=config['z_dim'], fe=config['fe'],
                                     fd=config['fd'], dr=config['dr'])
        if torch.cuda.device_count() > 1:
            self.vib = nn.DataParallel(self.vib)
        self.vib.to(self.device)
        self.optimizer = torch.optim.Adam(self.vib.parameters(),lr=config['lr'])

        if self.which_machine in self.is_cls:
            self.cls = classifier(config['z_dim'], num_models=num_sim)
            if torch.cuda.device_count() > 1:
                self.cls = nn.DataParallel(self.cls)
            self.cls.to(self.device)
            self.cls_optimizer = torch.optim.Adam(self.cls.parameters(), lr=1e-3)


        # Check point
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint_vib"))
            self.vig.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

            if self.which_machine in self.is_cls:
                model_state, optimizer_state = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint_cls"))
                self.cls.load_state_dict(model_state)
                cls_optimizer.load_state_dict(optimizer_state)

        self.callback_container.on_train_begin()

        # Start training
        for epoch in range(self.max_epoch):
            self.callback_container.on_epoch_begin(epoch)

            self.batch_accuracy = 0

            # Batch start
            for batch_idx, (X,y) in enumerate(self.train_loader):
                self.callback_container.on_batch_begin(batch_idx)
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                if self.which_machine in self.is_cls:
                    self.y_cls = y[:,-self.num_sim:]
                else:
                    self.y_cls = None
                self.y_param = y[:,:-self.num_sim]

                # VIB
                self.losses, self.pred =\
                        self.train_vib(X, self.y_param, config['beta'], config['gamma'],
                                       y_cls=self.y_cls)
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False

                # CLS
                if self.which_machine in self.is_cls:
                    cls_loss = self.train_cls(X,self.y_cls)

                self.callback_container.on_batch_end(batch_idx)

            # batch ended
            self.callback_container.on_epoch_end(epoch,
                                                 logs=self.log_container.history)

            """
            # Save models
            torch.save(self.vib.state_dict(),"./model/tmp/{}_{}_vib.pt".format(self.key,epoch))
            if self.which_machine in self.mach_vib_cls:
                torch.save(self.cls.state_dict(),"./model/tmp/{}_{}_cls.pt".format(self.key,epoch))

            new_best_epoch = self.log_container.history['best_epoch']

            # Delete models which do not fall into the convergence region
            if best_epoch != new_best_epoch:
                for i in range(best_epoch, new_best_epoch):
                    fmodel = "./model/tmp/{}_{}".format(self.key, i)
                    if os.path.exists(fmodel+'_vib.pt'):
                        os.remove(fmodel+'_vib.pt')
                    if self.which_machine in self.mach_vib_cls:
                        if os.path.exists(fmodel+'_cls.pt'):
                            os.remove(fmodel+'_cls.pt')
                best_epoch = new_best_epoch
                """


            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint_vib")
                torch.save((self.vib.state_dict(), self.optimizer.state_dict()),path)
                if self.which_machine in self.is_cls:
                    path = os.path.join(checkpoint_dir, "checkpoint_cls")
                    torch.save((self.cls.state_dict(), self.cls_optimizer.state_dict()),path)

            tune.report(loss=self.log_container.history['val_loss'][-1],
                        accuracy=self.log_container.history['val_accuray'][0])


            #if self.log_container.history['stop_training']:
            #    break

        # epoch ended 
        self.callback_container.on_train_end()
        print('Finished Training')

        """
        # Load the best model
        fmodel = "./model/tmp/{}_{}".format(self.key, best_epoch)
        self.vib.load_state_dict(torch.load(fmodel+'_vib.pt'))
        if self.which_machine in self.mach_vib_cls:
            self.cls.load_state_dict(torch.load(fmodel+'_cls.pt'))

        # Delete all tmp models
        for i in range(best_epoch, epoch+1):
            fmodel = "./model/tmp/{}_{}".format(self.key, i)
            if os.path.exists(fmodel+'_vib.pt'):
                os.remove(fmodel+'_vib.pt')
            if self.which_machine in self.mach_vib_cls:
                if os.path.exists(fmodel+'_cls.pt'):
                    os.remove(fmodel+'_cls.pt')
                    """


    def make_plots(self, epoch):
        ## Save Learning Curves
        fig = plt.figure(figsize=(30,20))
        colors = ['black', 'red', 'blue', 'green', 'yellow']
        lines  = ['-', '--', '-.', ':', '-']
        for i in range(self.num_loss):
            plt.plot(range(epoch+1), self.log_container[self.names_loss[i]],
                     label=self.names_loss[i],c=colors[i], ls=lines[i])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale("symlog")
        plt.legend()
        plt.savefig("./img/{}_loss_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save Accuracy
        fig = plt.figure(figsize=(30,20))
        plt.plot(range(epoch+1), self.log_container['accuracy'])
        plt.ylabel('Accuracy (relative error)')
        plt.xlabel('Epoch')
        plt.savefig("./img/{}_acc_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save AUC
        if self.which_machine in self.is_cls:
            fig = plt.figure(figsize=(30,20))
            plt.plot(range(epoch+1), self.log_container['auc_score'], label='Training',
                     c='k')
            plt.plot(range(epoch+1), self.log_container['auc_score_val'],
                     label="Valid", c='r')
            plt.legend()
            plt.ylabel('AUC score')
            plt.xlabel('Epoch')
            plt.savefig("./img/{}_auc_curve.png".format(self.fname),dpi=100)
            plt.close()



    def train_cls(self, X, y_cls):
        CE = nn.CrossEntropyLoss()
        # CLS Opt
        self.cls.train()
        self.cls.zero_grad()
        _, _     = self.vib(X)
        _mu      = self.vib.get_latent_variable()
        cls_pred = self.cls(_mu.detach())
        cls_loss = CE(y_cls, cls_pred)
        cls_loss.backward()
        self.cls_optimizer.step()
        self.cls.eval()
        return cls_loss


    def train_vib(self, X, y_true, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        y_pred, y_sigma = self.vib(X)
        _mu, _std       = self.vib.get_mu_std()
        if self.which_machine in self.is_cls:
            cls_pred        = self.cls(_mu.detach()) ## This is for score
            losses = vib_cls_loss(y_pred, y_sigma, y_true, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        else:
            losses = vib_loss(y_pred, y_sigma, y_true, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        losses[0].backward()
        self.optimizer.step()
        return losses, (y_pred, y_sigma, cls_pred)




