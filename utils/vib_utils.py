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
from copy import deepcopy

# Device Config
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
    J0 = torch.sum(torch.log(torch.mean(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    J1 = torch.sum((torch.mean(J1,axis=0)))
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return ((gamma*KL + J0+J1) / y.size(0), J0, J1, KL)


def dec_loss(y_mean,y_sigma, y):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.mean(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    J1 = torch.sum((torch.mean(J1,axis=0)))
    return J0+J1 / y.size(0), J0, J1

def cnn_enc_dec_loss(y_mean,y_sigma, y, y_class, _class, mu, std,
                     beta=0.01, gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.mean(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    #J1 = torch.sum(torch.mean(J1,axis=0))

    CrEn = nn.CrossEntropyLoss()
    CE   = CrEn(y_class, _class)
    KL   = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)/y.size(0)
    return (J0+J1-beta*CE+gamma*KL, J0, J1, -beta*CE, -gamma*KL)


def vib_cls_loss(y_mean,y_sigma, y, y_class, _class, mu, std, beta=0.01,
                 gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.mean(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.mean(J1,axis=0))
    J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))

    CrEn = nn.CrossEntropyLoss()
    #NL   = nn.NLLLoss()
    CE   = CrEn(y_class, _class)
    #CE   = NL(y_class, _class)
    KL   = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)/y.size(0)
    return (J0+J1-beta*CE+gamma*KL, J0, J1, -beta*CE, -gamma*KL)


def adaptive_vib_loss(y_mean, y_sigma, y, mu, std, gamma=0.01):
    J0   = torch.sum((y-y_mean)**2).mean()
    J1   = torch.sum(((y-y_mean)**2-y_sigma**2)**2).mean()
    KL   = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)/y.size(0)
    return (J0+J1+gamma*KL, J0, J1, gamma*KL)





class Trainer():


    def __init__(self, _vib, _cls, train_loader, which_machine="vib+cls",
                 valid_loader=None, fname='test', device='cuda',
          num_sim=1,verbose=True):

        self.train_loader  = train_loader
        self.valid_loader  = valid_loader
        self.fname         = fname
        self.device        = device
        self.which_machine = which_machine
        self.is_cls        = ['vib+cls', 'vib_cnn+cls','vib+cls_a',
                              'cnn_enc_dec']
        self.mach_vib      = ['vib', 'vib_cnn']
        self.mach_no_vib   = ['fcl', 'cnn']
        self.num_sim       = num_sim
        import secrets
        self.key = secrets.token_hex(4)


        if self.which_machine in self.mach_no_vib:
            self.num_loss  = 3
        elif self.which_machine in self.mach_vib:
            self.num_loss  = 4
        elif self.which_machine in self.is_cls:
            self.num_loss  = 5
        elif self.which_machine == 'avib':
            self.num_loss  = 4

        self.names_loss = ['total_loss', 'j0_loss', 'j1_loss', 'ce0_loss',
                           'ce1_loss']


        # Send to GPU if available
        #if torch.cuda.device_count() > 1:
             #_vib = nn.DataParallel(_vib)

        if which_machine in "separate_test":
            return
        self.vib = _vib.to(device)

        if which_machine in self.is_cls:
            #if torch.cuda.device_count() > 1:
            #    _cls = nn.DataParallel(_cls)
            self.cls = _cls.to(device)
        else:
            self.cls = None


    def run_cnn_separate(self,  encoder, decoder, classifier,
                         base_lr=1e-3, max_lr=1e-9,epochs=1000,
                         beta=1e-2, gamma=1e-2, patience=50, tol=1e-3, wd=1e-2, beta1=0.5,
                         beta2=0.999 ,verbose=True, 
                         save_plot=True, save_model=True):

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.cls     = classifier.to(self.device)

        self.enc_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=base_lr,
                                               weight_decay=wd,
                                               betas=(beta1, beta2))

        self.dec_optimizer = torch.optim.AdamW(self.decoder.parameters(),
                                              lr=1e-3)
        self.cls_optimizer = torch.optim.Adam(self.cls.parameters(),
                                              lr=1e-3)

        self.enc_scheduler = torch.optim.lr_scheduler.CyclicLR(self.enc_optimizer,
                                                      base_lr=base_lr,
                                                      max_lr=max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=500,
                                                      step_size_down=500)



        print('reday?')
        self.loss     = []
        self.accuracy = []
        self.auc      = []
        self.val_loss = []
        self.val_acc  = []
        self.val_auc  = []
        #########
        ## Epoch
        #########
        for epoch in range(epochs):
            epoch_start_time = time.time()

            mini_epochs    = 1
            epoch_accuracy = 0
            epoch_loss     = 0
            epoch_auc      = 0
            sample_seen    = 0
            y_cls_epoch    = []
            cls_pred_epoch = []
            z_epoch        = []

            #########
            ## Batch
            #########
            for batch_idx, (X,y) in enumerate(self.train_loader):
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                self.y_cls = y[:,-self.num_sim:]
                self.y_param = y[:,:-self.num_sim]

                #####
                # VIB
                #####
                self.losses, self.pred =\
                        self.train_enc_dec(X,self.y_param,beta,gamma,y_cls=self.y_cls)
                """
                for _ in range(mini_epochs):
                    self.losses_dec, self.pred_dec =\
                            self.train_dec(X,self.y_param)
                    """
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False

                #####
                # CLS
                #####
                cls_loss = self.train_cls_separate(X,self.y_cls)

                ###
                batch_size = X.size(0)
                epoch_loss = \
                        (sample_seen*epoch_loss +
                         self.losses[1].cpu().detach().numpy()*batch_size)\
                        /(sample_seen+batch_size)

                batch_accuracy = np.abs(self.y_param.cpu().detach().numpy()\
                        - self.pred[0].cpu().detach().numpy())\
                        /self.y_param.cpu().detach().numpy()
                epoch_accuracy = \
                        (sample_seen*epoch_accuracy +
                         np.sum(batch_accuracy,axis=0))\
                        /(sample_seen+batch_size)


                cls_pred = self.pred[2].detach().cpu().numpy()
                cls_pred[np.isnan(cls_pred)] = -1
                y_cls_epoch.append(self.y_cls.cpu().detach().numpy())
                cls_pred_epoch.append(cls_pred)

                sample_seen += batch_size
                del self.losses, self.pred, cls_pred

            first_batch_end  = time.time()

            """
            #########
            ## Batch for decoder
            #########
            epoch_loss_dec = 0
            epoch_acc_dec  = 0
            sample_seen    = 0
            for mini_epoch in range(mini_epochs):
                for batch_idx, (X,y) in enumerate(self.train_loader):
                    X       = X.to(self.device)
                    y       = y.float().to(self.device)
                    self.y_param = y[:,:-self.num_sim]
                    self.losses_dec, self.pred_dec =\
                            self.train_dec(X,self.y_param)
                    batch_size = X.size(0)
                    epoch_loss_dec = \
                            (sample_seen*epoch_loss_dec +
                             self.losses_dec[1].cpu().detach().numpy()*batch_size)\
                            /(sample_seen+batch_size)
                    batch_accuracy = np.abs(self.y_param.cpu().detach().numpy()\
                                            - self.pred_dec[0].cpu().detach().numpy())\
                            /self.y_param.cpu().detach().numpy()
                    epoch_acc_dec = \
                            (sample_seen*epoch_acc_dec +
                             np.sum(batch_accuracy,axis=0))\
                            /(sample_seen+batch_size)
                    sample_seen += batch_size
            del self.losses_dec, self.pred_dec

            second_batch_end  = time.time()
            """


            y_cls_epoch    = np.vstack(y_cls_epoch)
            cls_pred_epoch = np.vstack(cls_pred_epoch)
            epoch_auc = sklearn.metrics.roc_auc_score(y_cls_epoch,
                                                      cls_pred_epoch)

            val_loss, val_acc, val_auc= self.get_valid_loss_separate()
            first_batch_time  = round(first_batch_end - epoch_start_time)
            #second_batch_time = round(second_batch_end - first_batch_end)

            rnd = lambda x : round(float(x), 5)
            msg = f"epoch {epoch:<3}"
            msg += f" | {'loss':<3}: {rnd(epoch_loss):<8}"
            msg += f" | {'Om_m':<3}: {rnd(epoch_accuracy[0]):<8}"
            msg += f" | {'sig8':<3}: {rnd(epoch_accuracy[1]):<8}"
            #msg += f" | {'loss_2':<3}: {rnd(epoch_loss_dec):<8}"
            #msg += f" | {'Om_m_2':<3}: {rnd(epoch_acc_dec[0]):<8}"
            #msg += f" | {'sig8_2':<3}: {rnd(epoch_acc_dec[1]):<8}"
            msg += f" | {'auc':<3}: {rnd(epoch_auc):<8}"
            msg += f" |||  {'val_loss':<3}: {rnd(val_loss):<9}"
            msg += f" | {'val_Om_m':<3}: {rnd(val_acc[0]):<8}"
            msg += f" | {'val_sig8':<3}: {rnd(val_acc[1]):<8}"
            msg += f" | {'val auc':<3}: {rnd(val_auc):<8}"
            msg += f" | first batch: {first_batch_time:<8}"
            #msg += f" | second batch: {second_batch_time:<8}"
            print(msg)

            self.loss.append(epoch_loss)
            self.accuracy.append(np.mean(epoch_accuracy))
            self.auc.append(epoch_auc)

            self.val_loss.append(val_loss)
            self.val_acc.append(np.mean(val_acc))
            self.val_auc.append(val_auc)
            print(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
            print(torch.cuda.max_memory_allocated())

            if save_plot:
                self.make_plots_cnn(epoch)

        return self.encoder, self.decoder, self.cls, True



    def get_valid_loss_separate(self):
        n_val       = len(self.valid_loader.sampler)
        y_pred      = np.empty((n_val, 2),dtype=float)
        y_true      = np.empty((n_val, 2),dtype=float)
        y_std       = np.empty((n_val, 2),dtype=float)
        y_cls       = np.empty((n_val, self.num_sim),dtype=float)
        y_pred_cls  = np.empty((n_val, self.num_sim),dtype=float)
        start = 0
        with torch.no_grad():
            for N, (X_val, y_val) in enumerate(self.valid_loader):
                end                 = start+X_val.size(0)
                y_true[start:end,:], y_cls[start:end,:] =\
                        y_val[:,:self.num_sim], y_val[:,-self.num_sim:]
                if ~torch.is_tensor(X_val):
                    X_val = torch.tensor(X_val, dtype=torch.float).to(self.device)
                z                   = self.encoder(X_val)
                pred,    std        = self.decoder(z)
                y_pred[start:end,:] = pred.cpu().detach().numpy()
                y_std[start:end,:]  = std.cpu().detach().numpy()
                del pred, std
                if self.which_machine in self.is_cls:
                    pred_cls                = self.cls(z)
                    y_pred_cls[start:end,:] = pred_cls.cpu().detach().numpy()
                    del pred_cls
                start = end
                torch.cuda.empty_cache()

        diff     = y_true-y_pred
        val_acc  = np.mean(np.abs(diff)/y_true, axis=0)
        val_loss = np.mean(np.log(np.mean(diff**2, axis=0)))
        print(y_cls, y_pred_cls)
        val_auc  = sklearn.metrics.roc_auc_score(y_cls,y_pred_cls)
        return val_loss, val_acc, val_auc  




    def run_cnn(self, base_lr=1e-3, max_lr=1e-9,epochs=1000,
            beta=1e-2, gamma=1e-2, patience=50, tol=1e-3, wd=1e-2, beta1=0.5,
            beta2=0.999 ,verbose=True, 
            save_plot=True, save_model=True):

        num_micro_batches = self.train_loader.batch_size//10

        self.vib_optimizer = torch.optim.AdamW(self.vib.parameters(), lr=base_lr,
                                               weight_decay=wd,
                                               betas=(beta1, beta2))
        if self.which_machine in self.is_cls:
            self.cls_optimizer = torch.optim.Adam(self.cls.parameters(),
                                                  lr=1e-3)

        self.vib_scheduler = torch.optim.lr_scheduler.CyclicLR(self.vib_optimizer,
                                                      base_lr=base_lr,
                                                      max_lr=max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=500,
                                                      step_size_down=500)



        self.loss     = []
        self.accuracy = []
        self.auc      = []
        self.val_loss = []
        self.val_acc  = []
        self.val_auc  = []
        best_vib      = None
        best_cls      = None
        best_loss     = np.inf


        #########
        ## Epoch
        #########
        for epoch in range(epochs):
            epoch_start_time = time.time()

            epoch_accuracy = 0
            epoch_loss     = 0
            epoch_auc      = 0
            sample_seen    = 0
            y_cls_epoch    = []
            cls_pred_epoch = []

            #########
            ## Batch
            #########
            for batch_idx, (X,y) in enumerate(self.train_loader):
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                if self.which_machine in self.is_cls:
                    self.y_cls = y[:,-self.num_sim:]
                else:
                    self.y_cls = None
                self.y_param = y[:,:-self.num_sim]

                #####
                # VIB
                #####
                self.losses, self.pred =\
                        self.train_vib(X,self.y_param,beta,gamma,y_cls=self.y_cls)
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False

                #####
                # CLS
                #####
                if self.which_machine in self.is_cls:
                    cls_loss = self.train_cls(X,self.y_cls)

                batch_size = X.size(0)
                epoch_loss = \
                        (sample_seen*epoch_loss +
                         self.losses[1].cpu().detach().numpy()*batch_size)\
                        /(sample_seen+batch_size)

                batch_accuracy = np.abs(self.y_param.cpu().detach().numpy()\
                        - self.pred[0].cpu().detach().numpy())\
                        /self.y_param.cpu().detach().numpy()
                epoch_accuracy = \
                        (sample_seen*epoch_accuracy +
                         np.sum(batch_accuracy,axis=0))\
                        /(sample_seen+batch_size)

                if self.which_machine in self.is_cls:
                    cls_pred = self.pred[2].detach().cpu().numpy()
                    cls_pred[np.isnan(cls_pred)] = -1
                    y_cls_epoch.append(self.y_cls.cpu().detach().numpy())
                    cls_pred_epoch.append(cls_pred)

                sample_seen += batch_size
                del self.losses, self.pred


            if self.which_machine in self.is_cls:
                y_cls_epoch    = np.vstack(y_cls_epoch)
                cls_pred_epoch = np.vstack(cls_pred_epoch)
                epoch_auc = sklearn.metrics.roc_auc_score(y_cls_epoch,
                                                          cls_pred_epoch)


            val_loss, val_acc, val_auc= self.get_valid_loss()

            rnd = lambda x : round(float(x), 5)
            msg = f"epoch {epoch:<3}"
            msg += f" | {'loss':<3}: {rnd(epoch_loss):<8}"
            msg += f" | {'Om_m':<3}: {rnd(epoch_accuracy[0]):<8}"
            msg += f" | {'sig8':<3}: {rnd(epoch_accuracy[1]):<8}"
            if self.which_machine in self.is_cls:
                msg += f" | {'auc':<3}: {rnd(epoch_auc):<8}"
            msg += f" |||  {'val_loss':<3}: {rnd(val_loss):<9}"
            msg += f" | {'val_Om_m':<3}: {rnd(val_acc[0]):<8}"
            msg += f" | {'val_sig8':<3}: {rnd(val_acc[1]):<8}"
            if self.which_machine in self.is_cls:
                msg += f" | {'val auc':<3}: {rnd(val_auc):<8}"
            print(msg)


            if best_loss > val_loss:
                best_loss = val_loss
                best_vib  = self.vib
                if self.which_machine in self.is_cls:
                    best_cls  = self.cls

            self.loss.append(epoch_loss)
            self.accuracy.append(np.mean(epoch_accuracy))
            if self.which_machine in self.is_cls:
                self.auc.append(epoch_auc)
                self.val_acc.append(np.mean(val_acc))
                self.val_auc.append(val_auc)

            self.val_loss.append(val_loss)
            print(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
            print(torch.cuda.max_memory_allocated())

            if save_plot:
                self.make_plots_cnn(epoch)

        return best_vib, best_cls, True




    def make_plots_cnn(self, epoch):
        ## Save Learning Curves
        fig = plt.figure(figsize=(30,20))
        colors = ['black', 'red', 'blue', 'green', 'yellow']
        lines  = ['-', '--', '-.', ':', '-']

        plt.plot(range(epoch+1), self.loss,
                 label="Train",c=colors[0], ls=lines[0])
        plt.plot(range(epoch+1), self.val_loss,
                 label="Validation",c=colors[1], ls=lines[1])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale("symlog")
        plt.legend()
        plt.savefig("./img/{}_loss_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save Accuracy
        fig = plt.figure(figsize=(30,20))
        plt.plot(range(epoch+1), self.accuracy,
                 label="Train",c=colors[0], ls=lines[0])
        plt.plot(range(epoch+1), self.val_acc,
                 label="Validation",c=colors[1], ls=lines[1])
        plt.ylabel('Accuracy (relative error)')
        plt.xlabel('Epoch')
        plt.savefig("./img/{}_acc_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save AUC
        if self.which_machine in self.is_cls:
            fig = plt.figure(figsize=(30,20))
            plt.plot(range(epoch+1), self.auc,
                     label="Train",c=colors[0], ls=lines[0])
            plt.plot(range(epoch+1), self.val_auc,
                     label="Validation",c=colors[1], ls=lines[1])
            plt.legend()
            plt.ylabel('AUC score')
            plt.xlabel('Epoch')
            plt.savefig("./img/{}_auc_curve.png".format(self.fname),dpi=100)
            plt.close()



    def get_valid_loss(self):
        n_val       = len(self.valid_loader.sampler)
        y_pred      = np.empty((n_val, 2),dtype=float)
        y_true      = np.empty((n_val, 2),dtype=float)
        y_std       = np.empty((n_val, 2),dtype=float)
        y_cls       = np.empty((n_val, self.num_sim),dtype=float)
        y_pred_cls  = np.empty((n_val, self.num_sim),dtype=float)
        start = 0
        with torch.no_grad():
            for N, (X_val, y_val) in enumerate(self.valid_loader):
                end                 = start+X_val.size(0)
                y_true[start:end,:], y_cls[start:end,:] =\
                        y_val[:,:2], y_val[:,2:]
                if ~torch.is_tensor(X_val):
                    X_val = torch.tensor(X_val, dtype=torch.float).to(self.device)
                pred,    std        = self.vib(X_val)
                y_pred[start:end,:] = pred.cpu().detach().numpy()
                y_std[start:end,:]  = std.cpu().detach().numpy()
                del pred, std
                if self.which_machine in self.is_cls:
                    pred_cls                = self.cls(self.vib.get_latent_variable())
                    y_pred_cls[start:end,:] = pred_cls.cpu().detach().numpy()
                    del pred_cls
                start = end
                torch.cuda.empty_cache()

        diff     = y_true-y_pred
        val_acc  = np.mean(np.abs(diff)/y_true, axis=0)
        val_loss = np.mean(np.log(np.mean(diff**2, axis=0)))
        if self.which_machine in self.is_cls:
            val_auc  = sklearn.metrics.roc_auc_score(y_cls,y_pred_cls)
        else:
            val_auc  = None
        return val_loss, val_acc, val_auc  



    def run(self, base_lr=1e-3, max_lr=1e-9,epochs=1000,
            beta=1e-2, gamma=1e-2, patience=50, tol=1e-3, wd=1e-2, beta1=0.5,
            beta2=0.999 ,verbose=True, 
            save_plot=True, save_model=True):

        metric_name = []
        early_stopping_metric =  None
        best_epoch = 0
        self.log_container    = Log(metric_name,batch_size=self.train_loader.batch_size)
        self.early_stopping   = \
                EarlyStopping(early_stopping_metric, tol=tol, patience=patience)

        self.callback_container = CallbackContainer([self.log_container,])
                                                 #self.early_stopping])
        self.callback_container.set_trainer(self)

        start_time = time.time()

        # Optimiser RMSPROP
        #self.vib_optimizer = torch.optim.AdamW(self.vib.parameters(),
        #                                      lr=base_lr,)
        #self.vib_optimizer = torch.optim.AdamW(self.vib.parameters(), lr=base_lr,
                                               #weight_decay=wd,
                                               #betas=(beta1, beta2))
        self.vib_optimizer = torch.optim.NAdam(self.vib.parameters(), lr=base_lr)
        #self.vib_optimizer = torch.optim.Adadelta(self.vib.parameters(), lr=base_lr)
        #self.vib_optimizer = torch.optim.RMSprop(self.vib.parameters(), lr=base_lr)
        if self.which_machine in self.is_cls:
            self.cls_optimizer = torch.optim.Adam(self.cls.parameters(),
                                                  lr=1e-3)
                                                  #weight_decay=0.01)

        """
        self.vib_scheduler = torch.optim.lr_scheduler.CyclicLR(self.vib_optimizer,
                                                      base_lr=base_lr,
                                                      max_lr=max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=500,
                                                      step_size_down=500)
                                                      """

        self.vib_scheduler = None

        self.callback_container.on_train_begin()

        #########
        ## Epoch
        #########
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.callback_container.on_epoch_begin(epoch)

            self.batch_accuracy = 0
            if self.which_machine == 'avib':
                cls_loss_old = 0

            #########
            ## Batch
            #########
            for batch_idx, (X,y) in enumerate(self.train_loader):
                self.callback_container.on_batch_begin(batch_idx)
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                if self.which_machine in self.is_cls:
                    self.y_cls = y[:,-self.num_sim:]
                else:
                    self.y_cls = None
                self.y_param = y[:,:-self.num_sim]


                #####
                # VIB
                #####
                self.losses, self.pred =\
                        self.train_vib(X,self.y_param,beta,gamma,y_cls=self.y_cls)
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False



                #####
                # CLS
                #####
                if self.which_machine in self.is_cls:
                    cls_loss = self.train_cls(X,self.y_cls)


                if self.which_machine == "avib":
                    gamma += beta*(cls_loss-cls_loss_old)
                    gamma = min(1e-2, float(gamma.cpu().detach()))
                    gamma = max(1e3, gamma)
                    cls_loss_old = cls_loss

                self.callback_container.on_batch_end(batch_idx)

                #vib_scheduler.step()


            self.callback_container.on_epoch_end(epoch,
                                                 logs=self.log_container.history)

            """
            # Save models
            torch.save(self.vib.state_dict(),"./model/tmp/{}_{}_vib.pt".format(self.key,epoch))
            if self.which_machine in self.is_cls:
                torch.save(self.cls.state_dict(),"./model/tmp/{}_{}_cls.pt".format(self.key,epoch))
                """

            new_best_epoch = self.log_container.history['best_epoch']

            """
            # Delete models which do not fall into the convergence region
            if best_epoch != new_best_epoch:
                for i in range(best_epoch, new_best_epoch):
                    fmodel = "./model/tmp/{}_{}".format(self.key, i)
                    if os.path.exists(fmodel+'_vib.pt'):
                        os.remove(fmodel+'_vib.pt')
                    if self.which_machine in self.is_cls:
                        if os.path.exists(fmodel+'_cls.pt'):
                            os.remove(fmodel+'_cls.pt')
                best_epoch = new_best_epoch
                """


            #if self.log_container.history['stop_training']:
            #    break

            if save_plot:
                self.make_plots(epoch)

        """
        fmodel = "./model/tmp/{}_{}".format(self.key, best_epoch)
        self.vib.load_state_dict(torch.load(fmodel+'_vib.pt'))
        if self.which_machine in self.is_cls:
            self.cls.load_state_dict(torch.load(fmodel+'_cls.pt'))

        # Delete all tmp models
        for i in range(best_epoch, epoch+1):
            fmodel = "./model/tmp/{}_{}".format(self.key, i)
            if os.path.exists(fmodel+'_vib.pt'):
                os.remove(fmodel+'_vib.pt')
            if self.which_machine in self.is_cls:
                if os.path.exists(fmodel+'_cls.pt'):
                    os.remove(fmodel+'_cls.pt')
                    """

        #self.callback_container.on_train_end()

        return self.vib, self.cls, True


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



    def train_enc_dec(self, X, y_true, beta, gamma, y_cls=None):

        self.encoder.train()
        self.encoder.zero_grad()
        self.decoder.train()
        self.decoder.zero_grad()

        z               = self.encoder(X)
        y_pred, y_sigma = self.decoder(z)

        self.cls.eval()
        self.cls.requires_grad = False
        _mu, _std = self.encoder.get_mu_std()
        cls_pred  = self.cls(_mu.detach()) ## This is for score
        losses    = vib_cls_loss(y_pred, y_sigma, y_true, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        losses[0].backward()
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        self.enc_scheduler.step()
        del _mu, _std 
        return losses, (y_pred, y_sigma, cls_pred)


    def train_dec(self, X, y_true):
        self.cls.requires_grad = False
        self.encoder.requires_grad = False

        self.decoder.train()
        self.decoder.zero_grad()

        z               = self.encoder(X)
        y_pred, y_sigma = self.decoder(z)

        losses    = dec_loss(y_pred, y_sigma, y_true)
        losses[0].backward()
        self.dec_optimizer.step()
        return losses, (y_pred, y_sigma)


    def train_cls_separate(self, X, y_cls):
        CE = nn.CrossEntropyLoss()
        # CLS Opt
        self.encoder.eval()
        self.encoder.requires_grad = False
        self.cls.train()
        self.cls.zero_grad()
        z        = self.encoder(X)
        cls_pred = self.cls(z.detach())
        cls_loss = CE(y_cls, cls_pred)
        cls_loss.backward()
        self.cls_optimizer.step()
        return cls_loss


    def train_cls(self, X, y_cls):
        CE = nn.CrossEntropyLoss()
        # CLS Opt
        self.vib.eval()
        self.vib.requires_grad = False
        self.cls.train()
        self.cls.zero_grad()
        _, _     = self.vib(X)
        _z      = self.vib.get_latent_variable()
        cls_pred = self.cls(_z.detach())
        cls_loss = CE(y_cls, cls_pred)
        cls_loss.backward()
        self.cls_optimizer.step()
        return cls_loss


    def train_vib(self, X, y_true, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        y_pred, y_sigma = self.vib(X)
        if self.which_machine in self.mach_no_vib:
            losses   = no_vib_loss(y_pred, y_sigma, y_true)
            cls_pred =  None 
        elif self.which_machine in self.mach_vib:
            _mu, _std = self.vib.get_mu_std()
            losses    = vib_loss(y_pred, y_sigma, y_true, _mu, _std, gamma=gamma)
            cls_pred  =  None 
        elif self.which_machine in self.is_cls:
            self.cls.eval()
            self.cls.requires_grad = False
            _mu, _std = self.vib.get_mu_std()
            #print(_mu.size(),_std.size())
            cls_pred  = self.cls(_mu.detach()) ## This is for score
            losses    = vib_cls_loss(y_pred, y_sigma, y_true, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        losses[0].backward()
        self.vib_optimizer.step()
        if self.vib_scheduler is not None:
            self.vib_scheduler.step()
        del _mu, _std 
        return losses, (y_pred, y_sigma, cls_pred)

    def train_cnn_enc_dec(self, X, y_true, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        self.cls.eval()
        self.cls.requires_grad = False


        y_mean, y_std = self.vib(X)
        _mu, _std       = self.vib.get_mu_std()
        #z1              = self.get_cnn_summary()
        cls_pred        = self.cls(_mu.detach()) ## This is for score
        losses    = cnn_enc_dec_loss(y_mena, y_std, y_true, y_cls,
                                 cls_pred.detach(), _mu, _std,
                                 beta=beta, gamma=gamma)
        losses[0].backward()
        self.vib_optimizer.step()
        if self.vib_scheduler is not None:
            self.vib_scheduler.step()
        del _mu, _std
        return losses, (y_pred, y_sigma, cls_pred)


