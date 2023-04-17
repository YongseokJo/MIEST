import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
import sklearn
from torch.utils.data.sampler import SubsetRandomSampler

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu' # temporarily
# Fix random seeds for reproducibility



class VIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""
    def __init__(self, input_shape, output_shape, z_dim, num_models=2, h=1, dr=0.5):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(VIB,self).__init__()

        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim 
        self.num_models     = num_models

        self.nn_encoder = nn.Sequential(
            nn.Linear(self.input_shape,1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024,512),
        )

        self.nn_weights  = nn.Linear(512, self.z_shape) 
        self.nn_std   = nn.Linear(512, self.z_shape)

        self.nn_decoder_mean = nn.Sequential(nn.Linear(self.z_shape, 128),
                                        nn.GELU(),
                                        nn.LayerNorm(128),
                                        nn.Linear(128, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        nn.Linear(32, self.output_shape))
        self.nn_decoder_sigma = nn.Sequential(nn.Linear(self.z_shape, 128),
                                        nn.GELU(),
                                        nn.LayerNorm(128),
                                        nn.Linear(128, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        nn.Linear(32, self.output_shape))

    def encoder(self, x):
        """
        x : (input_shape)
        """
        x = self.nn_encoder(x)
        #return self.nn_token(x), F.softplus(self.nn_prob(x)-5, beta=1)
        return self.nn_weights(x), F.softplus(self.nn_weights(x)-5, beta=1)

    def decoder(self, z):
        """
        z : (candidate_size)
        """
        return self.nn_decoder_mean(z), self.nn_decoder_sigma(z) 

    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps

    def get_latent_variable(self):
        return self.z

    def get_mu_std(self):
        return self.mu, self.std

    def forward(self, x):
        """
        Forward pass 
        Parameters:
        -----------
        x : (input_shape)
        """
        self.mu, self.std = self.encoder(x)
        self.z = self.reparameterise(self.mu, self.std)
        return self.decoder(self.z)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class linearVIB(VIB):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(linearVIB,self).__init__()

        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2

        self.nn_weights  = nn.Linear(self.input_shape, self.z_shape) 
        self.nn_std   = nn.Linear(self.input_shape, self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        #nn.Linear(64, 16),
                                        #nn.GELU(),
                                        #nn.LayerNorm(16),
                                        nn.Linear(32, self.output_shape))

class polyVIB(VIB):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim, degree):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(polyVIB,self).__init__()
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        self.degree         = degree

        self.nn_weights  = nn.Linear(self.input_shape*len(self.degree), self.z_shape) 
        self.nn_std      = nn.Linear(self.input_shape*len(self.degree), self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 128),
                                        nn.GELU(),
                                        nn.LayerNorm(128),
                                        nn.Linear(128, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        nn.Linear(32, self.output_shape))


class VIB_a(VIB):
    """An implementation of the Variational Information Bottleneck Method."""
    def __init__(self, input_shape, output_shape, z_dim, num_models=2, h=1, dr=0.5):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(VIB_a,self).__init__(input_shape, output_shape, z_dim)

        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim
        self.num_models     = num_models

        self.nn_encoder = nn.Sequential(
            nn.Linear(self.input_shape,int(self.input_shape/2*h)),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Linear(int(self.input_shape/2*h),int(self.input_shape/4*h)),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Linear(int(self.input_shape/4*h),int(self.input_shape/8*h)),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Linear(int(self.input_shape/8*h), int(self.z_shape))
        )

        self.nn_decoder = nn.Sequential(
            nn.Linear(self.z_shape, int(self.z_shape/2)),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Linear(int(self.z_shape/2), int(self.z_shape/4)),
            nn.LeakyReLU(),
            nn.Dropout(dr),
            nn.Linear(int(self.z_shape/4), int(self.output_shape*2)))


    def encoder(self, x):
        mean = self.nn_encoder(x)
        std  = F.softplus(mean-5,beta=1)
        return mean, std

    def decoder(self, z):
        mean, std = self.nn_decoder(z)[:,self.output_shape:], self.nn_decoder(z)[:,:self.output_shape]
        return mean, std

    
    
class model_o3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_o3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y




class classifier(nn.Module):

    def __init__(self, z_dim, num_models=2):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(classifier,self).__init__()

        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        self.num_models     = num_models

        self.nn_classifier = nn.Sequential(nn.Linear(self.z_shape, 128),
                                           nn.GELU(),
                                           nn.LayerNorm(128),
                                           nn.Linear(128, 32),
                                           nn.GELU(),
                                           nn.LayerNorm(32),
                                           nn.Linear(32, self.num_models),
                                           nn.Softmax(dim=1)
                                          )


    def get_class(self):
        return self._class

    def forward(self, x):
        return self.nn_classifier(x)


###################
### Loss functions
###################
def vib_loss(y_mean,y_sigma, y, mu, std, gamma=0.01):
    J0 = torch.sum((y-y_mean)**2).sum()
    J1 = torch.sqrt(torch.sum(((y-y_mean)**2-y_sigma**2)**2).sum())
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return ((gamma*KL + J0+J1) / y.size(0), J0, J1, KL)


def vib_cls_loss(y_mean,y_sigma, y, y_class, _class, mu, std, beta=0.01,
                 gamma=0.01):
    J0   = torch.sum((y-y_mean)**2).mean()
    J1   = torch.sqrt(torch.sum(((y-y_mean)**2-y_sigma**2)**2).mean())
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
    def __init__(self, _vib, _cls, train_loader, which_machine="vib+cls", test_dataset=None, fname='test', device='cuda',
          verbose=True):

        self.train_loader  = train_loader
        self.test_dataset  = test_dataset
        self.fname         = fname
        self.device        = device
        self.which_machine = which_machine

        # Send to GPU if available
        self.vib = _vib.to(device)
        if which_machine == 'vib':
            self.cls = None
        else:
            self.cls = _cls.to(device)


        num_losses = {"vib":4, "vib+cls":5, "vib+cls_a":5, "avib":4}
        self.num_loss   = num_losses[which_machine]
        self.names_loss = ['total_loss', 'j0_loss', 'j1_loss', 'ce0_loss',
                           'ce1_loss']

    def run(self, learning_rate=1e-3, epochs=1000, decay_rate=None,
            beta=1e-2, gamma=1e-2, verbose=True,
            save_plot=True, save_model=True):

        measures = defaultdict(list)
        start_time = time.time()

        # Optimiser
        self.vib_optimiser = torch.optim.Adam(self.vib.parameters(), lr=learning_rate)
        if self.which_machine != 'vib':
            self.cls_optimiser = torch.optim.Adam(self.cls.parameters(), lr=learning_rate) #optimiser = torch.optim.SGD(vib.parameters(), lr=learning_rate)

        if decay_rate is not None:
            vib_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.vib_optimiser, gamma=decay_rate)
            if self.which_machine != 'vib':
                cls_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.cls_optimiser, gamma=decay_rate)


        #########
        ## Epoch
        #########
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # exponential decay of learning rate every 2 epochs
            if epoch % 10 == 0 and epoch > 0 and decay_rate is not None:
                vib_scheduler.step()
                if self.which_machine != 'vib':
                    cls_scheduler.step()


            batch_losses   = [0]*self.num_loss
            batch_accuracy = 0
            if self.which_machine == 'avib':
                cls_loss_old = 0

            #########
            ## Batch
            #########
            for _, (X,y) in enumerate(self.train_loader):
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                if self.which_machine != 'vib':
                    y_cls = y[:,2:]
                else:
                    y_cls = None
                y_param = y[:,:2]


                #####
                # VIB
                #####
                losses, pred = self.train_vib(X,y_param,beta,gamma,y_cls=y_cls)
                if torch.isnan(losses[0]) or torch.isinf(losses[0]):
                    return self.vib, self.cls, False

                for i in range(self.num_loss):
                    batch_losses[i] += losses[i].item()*X.size(0)
                batch_accuracy += \
                        torch.sum(torch.abs(y_param - pred[0])/y_param)/X.size(0)


                #####
                # CLS
                #####
                if self.which_machine != "vib":
                    cls_loss = self.train_cls(X,y_cls)


                if self.which_machine == "avib":
                    gamma += beta*(cls_loss-cls_loss_old)
                    gamma = min(1e-2, float(gamma.cpu().detach()))
                    gamma = max(1e3, gamma)
                    cls_loss_old = cls_loss



            if self.test_dataset is not None:
                X_test, y_test = self.test_dataset.tensors
                y_param        = y_test[:,:2]
                y_pred, _      = self.vib(X_test.to(self.device))
                y_pred         = y_pred.detach().cpu().numpy()
                test_accuray   = np.abs(y_param-y_pred)/y_param
                test_accuray   = test_accuray.mean(axis=0)

                if self.which_machine != 'vib':
                    cls_test       = y_test[:,2:]
                    _mu_test, _    = self.vib.get_mu_std()
                    cls_test_pred  = self.cls(_mu_test.detach())
            else:
                test_accuray = -1.0

            # Save losses per epoch
            for i in range(self.num_loss):
                measures[self.names_loss[i]].append(batch_losses[i]/len(self.train_loader.dataset)) 


            if self.which_machine != 'vib':
                cls_pred = pred[2].detach().cpu().numpy()
                cls_pred[np.isnan(cls_pred)] = -1
                measures['auc_score'].append(sklearn.metrics.roc_auc_score(
                    y_cls.clone().detach().cpu().numpy(),cls_pred))
                cls_test_pred = cls_test_pred.detach().cpu().numpy()
                cls_test_pred[np.isnan(cls_test_pred)] = -1
                measures['auc_score_test'].append(sklearn.metrics.roc_auc_score(
                    cls_test.clone().detach().cpu().numpy(),cls_test_pred))
            measures['accuracy'].append(batch_accuracy.cpu().detach() /\
                                        len(self.train_loader.dataset))
            self.measures = measures


            # Save models
            if save_model:
                torch.save(self.vib.state_dict(),"./model/{}_vib.pt".format(self.fname))
                if self.which_machine != 'vib':
                    torch.save(self.cls.state_dict(),"./model/{}_cls.pt".format(self.fname))

            if save_plot:
                self.make_plots(epoch)

            #if (epoch + 1) % 100 == 0 and epoch > 0 and verbose:
            if verbose:
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Loss: {:.4f}...".format(measures['total_loss'][-1]),
                      "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
                      "Test Om: {:.3f} sig: {:.3f}".format(test_accuray[0],test_accuray[1]))
                #"Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))
        return self.vib, self.cls, True


    def make_plots(self, epoch):
        ## Save Learning Curves
        fig = plt.figure(figsize=(30,20))
        colors = ['black', 'red', 'blue', 'green', 'yellow']
        lines  = ['-', '--', '-.', ':', '-']
        for i in range(self.num_loss):
            plt.plot(range(epoch+1), self.measures[self.names_loss[i]],
                     label=self.names_loss[i],c=colors[i], ls=lines[i])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale("symlog")
        plt.legend()
        plt.savefig("./img/{}_loss_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save Accuracy
        fig = plt.figure(figsize=(30,20))
        plt.plot(range(epoch+1), self.measures['accuracy'])
        plt.ylabel('Accuracy (relative error)')
        plt.xlabel('Epoch')
        plt.savefig("./img/{}_acc_curve.png".format(self.fname),dpi=100)
        plt.close()

        ## Save AUC
        if self.which_machine != 'vib':
            fig = plt.figure(figsize=(30,20))
            plt.plot(range(epoch+1), self.measures['auc_score'], label='Training',
                     c='k')
            plt.plot(range(epoch+1), self.measures['auc_score_test'], label="Test", c='r')
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
        self.cls_optimiser.step()
        self.cls.eval()
        return cls_loss


    def train_vib(self, X, y, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        y_pred, y_sigma = self.vib(X)
        _mu, _std       = self.vib.get_mu_std()
        if self.which_machine == 'vib':
            losses = vib_loss(y_pred, y_sigma, y, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        elif self.which_machine == 'vib+cls':
            cls_pred        = self.cls(_mu.detach()) ## This is for score
            losses = vib_cls_loss(y_pred, y_sigma, y, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        elif self.which_machine == 'vib+cls_a':
            cls_pred        = self.cls(_mu.detach()) ## This is for score
            losses = vib_cls_loss(y_pred, y_sigma, y, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        elif self.which_machine == 'avib':
            losses = adaptive_vib_loss(y_pred,y_sigma, y, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        losses[0].backward()
        self.vib_optimiser.step()
        return losses, (y_pred, y_sigma, cls_pred)









