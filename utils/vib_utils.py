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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu' # temporarily
# Fix random seeds for reproducibility



class FCL(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(linearVIB,self).__init__()

        self.input_shape    = input_shape
        self.output_shape   = output_shape

        self.fcl = nn.Sequential(nn.Linear(self.z_shape, 1024),
                                        nn.GELU(),
                                        nn.LayerNorm(1024),
                                        nn.Linear(1024, 256),
                                        nn.GELU(),
                                        nn.LayerNorm(256),
                                        nn.Linear(256, 64),
                                        nn.GELU(),
                                        nn.LayerNorm(64),
                                        nn.Linear(64, self.output_shape)
                                       )
        def forward(self, x):
            x = fcl(x)
            Y = torch.clone(x)
            return y[:,:2], torch.square(y[:,2:])



class VIB_old(nn.Module):
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


class VIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""
    def __init__(self, input_shape, output_shape, z_dim, fe=0.5, fd=0.25, dr=0.5):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(VIB,self).__init__()

        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_dim          = z_dim 
        encoder_max_depth   = 10
        decoder_max_depth   = 4

        self.encoder = nn.ModuleList([])
        in_ = self.input_shape
        for i in range(encoder_max_depth):
            out_ = int(in_*fe)
            if out_ < z_dim or i == encoder_max_depth-1:
                break
            self.encoder.extend(
                [
                    nn.Linear(in_,out_),
                    nn.GELU(),
                    nn.Dropout(dr),
                    nn.LayerNorm(out_),
                ]
            )
            in_ = out_

        self.encoder.append(nn.Linear(in_,z_dim))


        self.decoder = nn.ModuleList([])
        in_ = z_dim
        for i in range(decoder_max_depth):
            out_ = int(in_*fd)
            if out_ < int(output_shape*2) or i == decoder_max_depth-1:
                break
            self.decoder.extend(
                [
                    nn.Linear(in_,out_),
                    nn.GELU(),
                    nn.Dropout(dr),
                    nn.LayerNorm(out_),
                ]
            )
            in_ = out_

        self.decoder.append(nn.Linear(in_,int(2*output_shape)))


    def forward_encoder(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x, F.softplus(x-5, beta=1)

    def forward_decoder(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x[:,self.output_shape:], x[:,:self.output_shape]

    def reparameterise(self, mean, std):
        eps = torch.randn_like(std)
        return mean + std*eps

    def get_latent_variable(self):
        return self.z

    def get_mu_std(self):
        return self.mu, self.std

    def forward(self, x):
        self.mu, self.std = self.forward_encoder(x)
        self.z = self.reparameterise(self.mu, self.std)
        return self.forward_decoder(self.z)

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



# From Paco's training code
class CNN(nn.Module):
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





class VIB_CNN(nn.Module):
    def __init__(self, hidden, dr, channels, outputs=2):
        super(VIB_CNN, self).__init__()
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  math.ceil(2*hidden), kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(math.ceil(2*hidden),  math.ceil(2*hidden), kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(math.ceil(2*hidden),  math.ceil(2*hidden), kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(math.ceil(2*hidden))
        self.B02 = nn.BatchNorm2d(math.ceil(2*hidden))
        self.B03 = nn.BatchNorm2d(math.ceil(2*hidden))

        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(math.ceil(2*hidden), math.ceil(4*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(math.ceil(4*hidden), math.ceil(4*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(math.ceil(4*hidden), math.ceil(4*hidden), kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(math.ceil(4*hidden))
        self.B12 = nn.BatchNorm2d(math.ceil(4*hidden))
        self.B13 = nn.BatchNorm2d(math.ceil(4*hidden))

        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(math.ceil(4*hidden), math.ceil(8*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(math.ceil(8*hidden), math.ceil(8*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(math.ceil(8*hidden), math.ceil(8*hidden), kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(math.ceil(8*hidden))
        self.B22 = nn.BatchNorm2d(math.ceil(8*hidden))
        self.B23 = nn.BatchNorm2d(math.ceil(8*hidden))

        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(math.ceil(8*hidden),  math.ceil(16*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(math.ceil(16*hidden), math.ceil(16*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(math.ceil(16*hidden), math.ceil(16*hidden), kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(math.ceil(16*hidden))
        self.B32 = nn.BatchNorm2d(math.ceil(16*hidden))
        self.B33 = nn.BatchNorm2d(math.ceil(16*hidden))

        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(math.ceil(16*hidden), math.ceil(32*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(math.ceil(32*hidden), math.ceil(32*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(math.ceil(32*hidden), math.ceil(32*hidden), kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(math.ceil(32*hidden))
        self.B42 = nn.BatchNorm2d(math.ceil(32*hidden))
        self.B43 = nn.BatchNorm2d(math.ceil(32*hidden))

        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(math.ceil(32*hidden), math.ceil(64*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(math.ceil(64*hidden), math.ceil(64*hidden), kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(math.ceil(64*hidden), math.ceil(64*hidden), kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(math.ceil(64*hidden))
        self.B52 = nn.BatchNorm2d(math.ceil(64*hidden))
        self.B53 = nn.BatchNorm2d(math.ceil(64*hidden))

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(math.ceil(64*hidden), math.ceil(128*hidden), kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(math.ceil(128*hidden))

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(math.ceil(128*hidden), math.ceil(64*hidden))
        self.FC2  = nn.Linear(math.ceil(64*hidden),  math.ceil(2*outputs))

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


    def encoder(self, image):
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
        return x, F.softplus(x-5, beta=1)

    def decoder(self, x):
        #x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,2:4] = torch.square(x[:,2:4])
        return y[:,:2], y[:,2:]

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
def no_vib_loss(y_mean,y_sigma, y):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.sum((torch.sum(J1,axis=0)))
    return J0+J1 / y.size(0), J0, J1


def vib_loss(y_mean,y_sigma, y, mu, std, gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.sum((torch.sum(J1,axis=0)))
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return ((gamma*KL + J0+J1) / y.size(0), J0, J1, KL)


def vib_cls_loss(y_mean,y_sigma, y, y_class, _class, mu, std, beta=0.01,
                 gamma=0.01):
    J0 = (y-y_mean)**2
    J0 = torch.sum(torch.log(torch.sum(J0,axis=0)))
    J1 = ((y-y_mean)**2-y_sigma**2)**2
    #J1 = torch.sum(torch.log(torch.sum(J1,axis=0)))
    J1 = torch.sum((torch.sum(J1,axis=0)))
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
    def __init__(self, _vib, _cls, train_loader, which_machine="vib+cls", val_dataset=None, fname='test', device='cuda',
          verbose=True):

        self.train_loader  = train_loader
        self.val_dataset   = val_dataset
        self.fname         = fname
        self.device        = device
        self.which_machine = which_machine
        self.mach_vib_cls  = ['vib+cls', 'vib+cls_a']
        self.mach_vib      = ['vib', 'vib_cnn']
        self.mach_no_vib   = ['fcl', 'cnn']


        if self.which_machine in self.mach_no_vib:
            self.num_loss  = 3
        elif self.which_machine in self.mach_vib:
            self.num_loss  = 4
        elif self.which_machine in self.mach_vib_cls:
            self.num_loss  = 5
        elif self.which_machine == 'avib':
            self.num_loss  = 4

        self.names_loss = ['total_loss', 'j0_loss', 'j1_loss', 'ce0_loss',
                           'ce1_loss']


        # Send to GPU if available
        self.vib = _vib.to(device)
        if which_machine in self.mach_vib_cls:
            self.cls = _cls.to(device)
        else:
            self.cls = None



    def run(self, learning_rate=1e-3, epochs=1000, decay_rate=None,
            beta=1e-2, gamma=1e-2, patience=50, tol=1e-3,verbose=True, 
            save_plot=True, save_model=True):

        metric_name = []
        early_stopping_metric =  None
        self.log_container    = Log(metric_name,batch_size=self.train_loader.batch_size)
        self.early_stopping   = \
                EarlyStopping(early_stopping_metric, tol=tol, patience=patience)

        self.callback_container = CallbackContainer([self.log_container,
                                                 self.early_stopping])
        self.callback_container.set_trainer(self)

        start_time = time.time()

        # Optimiser RMSPROP
        self.vib_optimiser = torch.optim.Adam(self.vib.parameters(),
                                              lr=learning_rate,)
                                              #weight_decay=0.01)
                                              #cycle_moment=True)
        if self.which_machine in self.mach_vib_cls:
            self.cls_optimiser = torch.optim.Adam(self.cls.parameters(),
                                                  lr=1e-3)
                                                  #weight_decay=0.01)

        """
        vib_scheduler =\
                torch.optim.lr_scheduler.CyclicLR(self.vib_optimiser,
                                                  base_lr=learning_rate,
                                                  max_lr=0.01,
                                                  step_size_up=2000,mode="triangular2",
                                                 )
                                                 """

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
                if self.which_machine in self.mach_vib_cls:
                    self.y_cls = y[:,2:]
                else:
                    self.y_cls = None
                y_param = y[:,:2]


                #####
                # VIB
                #####
                self.losses, self.pred =\
                        self.train_vib(X,y_param,beta,gamma,y_cls=self.y_cls)
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False

                self.batch_accuracy += \
                        torch.sum(torch.abs(y_param - self.pred[0])/y_param)/X.size(0)


                #####
                # CLS
                #####
                if self.which_machine in self.mach_vib_cls:
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

            if self.log_container.history['best_epoch'] == epoch:
                vib_best_weights = deepcopy(self.vib.state_dict())
                if self.which_machine in self.mach_vib_cls:
                    cls_best_weights = deepcopy(self.cls.state_dict())

            if self.log_container.history['stop_training']:
                break

            # Save models
            if save_model:
                torch.save(self.vib.state_dict(),"./model/{}_vib.pt".format(self.fname))
                if self.which_machine in self.mach_vib_cls:
                    torch.save(self.cls.state_dict(),"./model/{}_cls.pt".format(self.fname))

            if save_plot:
                self.make_plots(epoch)


        self.callback_container.on_train_end()

        self.vib.load_state_dict(vib_best_weights)
        if self.which_machine in self.mach_vib_cls:
            self.cls.load_state_dict(cls_best_weights)

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
        if self.which_machine in self.mach_vib_cls:
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
        self.cls_optimiser.step()
        self.cls.eval()
        return cls_loss


    def train_vib(self, X, y_true, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        y_pred, y_sigma = self.vib(X)
        _mu, _std       = self.vib.get_mu_std()
        if self.which_machine in self.mach_no_vib:
            losses = no_vib_loss(y_pred, y_sigma, y_true)
            cls_pred        =  None 
        elif self.which_machine in self.mach_vib:
            losses = vib_loss(y_pred, y_sigma, y_true, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        elif self.which_machine in self.mach_vib_cls:
            cls_pred        = self.cls(_mu.detach()) ## This is for score
            losses = vib_cls_loss(y_pred, y_sigma, y_true, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        elif self.which_machine == 'avib':
            losses = adaptive_vib_loss(y_pred,y_sigma, y_true, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        losses[0].backward()
        self.vib_optimiser.step()
        return losses, (y_pred, y_sigma, cls_pred)




