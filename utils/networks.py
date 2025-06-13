import torch
import torch.nn as nn
from torch.nn import functional as F
import math


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


class opFCL(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, f=0.5, dr=0.5):
        super(linearVIB,self).__init__()

        if f > 1.0 or f < 0.0:
            print('f should be equal to or less than 1, setting f to 1')
            f = np.max(0, np.min(1, f))

        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.z_dim        = z_dim 
        _max_depth        = 10

        self.fcl = nn.ModuleList([])
        in_ = self.input_shape
        for i in range(encoder_max_depth):
            out_ = int(in_*f)
            if out_ < 2*output_shape or i == encoder_max_depth-1:
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

        self.encoder.append(nn.Linear(in_,2*output_shape))

        def forward(self, x):
            x = fcl(x)
            return x[:,:output_shape], torch.square(x[:,self.output_shape:])


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

        if fd > 1.0 or fe > 1.0:
            print('fd and fe should be equal to or less than 1')
            raise
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_dim          = z_dim
        encoder_max_depth   = 20
        decoder_max_depth   = 20

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
        in_        = z_dim
        self.z_dim = z_dim
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

    def get_z_dim(self):
        return self.z_dim

    def forward(self, x):
        self.mu, self.std = self.forward_encoder(x)
        self.z = self.reparameterise(self.mu, self.std)
        return self.forward_decoder(self.z)
        #return self.forward_decoder(self.mu)

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
        super(CNN, self).__init__()
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

        #self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)
        self.FC2  = nn.Linear(64*hidden,  4)

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

    def _init_weights(self):
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
        y[:,2:] = torch.square(x[:,2:])

        return y[:,:2], y[:,2:]





class VIB_CNN(nn.Module):
    def __init__(self, hidden, dr, channels, z_dim,outputs=2):
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

        self.z_dim  =\
                math.ceil(max(min(math.ceil(128*hidden),z_dim),math.ceil(32*hidden)))
        self.FC1  = nn.Linear(math.ceil(128*hidden), self.z_dim*2)

        self.FC2  = nn.Linear(self.z_dim,  math.ceil(32*hidden))
        self.FC3  = nn.Linear(math.ceil(32*hidden),  math.ceil(8*hidden))
        self.FC4  = nn.Linear(math.ceil(8*hidden),  math.ceil(2*outputs))

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

    def _init_weights(self):
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
        x = self.dropout(x)
        x = self.FC1(x)

        return x[:,:self.z_dim], F.softplus(x[:,self.z_dim:]-5, beta=1)

    def decoder(self, x):
        #x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.dropout(self.LeakyReLU(self.FC3(x)))
        x = self.FC4(x)

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

    def get_z_dim(self):
        return self.z_dim

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






class cnn_encoder_decoder(nn.Module):
    def __init__(self, hidden, dr, channels, z_dim,outputs=2):
        super(cnn_encoder_decoder, self).__init__()
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


        # input: 16*hiddenx16x16 ----------> output: 4096*hiddenx1x1
        #self.z1       = math.ceil(4096*hidden)
        #self.C61 = nn.Conv2d(math.ceil(16*hidden), self.z1, kernel_size=16, stride=1, padding=0,
        #                    padding_mode='circular', bias=True)
        #self.B61 = nn.BatchNorm2d(self.z1)

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
        self.z1 = math.ceil(128*4*hidden)
        self.C61 = nn.Conv2d(math.ceil(64*hidden), self.z1, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(self.z1)


        # encoder
        self.z_dim  =\
                math.ceil(min(self.z1, z_dim))
        self.FC1  = nn.Linear(self.z1, int(self.z1/2))
        self.FC2  = nn.Linear(int(self.z1/2), int(self.z1/4))
        self.FC3  = nn.Linear(int(self.z1/4), int(self.z_dim*2))

        # decoder
        self.FC4  = nn.Linear(self.z_dim, int(self.z_dim/4))
        self.FC5  = nn.Linear(int(self.z_dim/4),  int(self.z_dim/16))
        self.FC6  = nn.Linear(int(self.z_dim/16),  math.ceil(2*outputs))

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def cnn(self, image):
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
        #x = self.LeakyReLU(self.B61(self.C61(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        return x

    def encoder(self, x):
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.FC3(x)
        return x[:,:self.z_dim], F.softplus(x[:,self.z_dim:]-5, beta=1)

    def decoder(self, x):
        x = self.dropout(self.LeakyReLU(self.FC4(x)))
        x = self.dropout(self.LeakyReLU(self.FC5(x)))
        x = self.FC6(x)

        # enforce the errors to be positive
        #y = torch.clone(x)
        #y[:,2:] = torch.square(y[:,2:])
        return x[:,:2], torch.square(x[:,2:])

    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps

    def get_cnn_summary(self):
        return self.z_cnn

    def get_latent_variable(self):
        return self.z

    def get_mu_std(self):
        return self.mu, self.std

    def get_z_dim(self):
        return self.z_dim

    def forward(self, x):
        """
        Forward pass 
        Parameters:
        -----------
        x : (input_shape)
        """
        self.z_cnn = self.cnn(x)
        self.mu, self.std = self.encoder(self.z_cnn)
        self.z = self.reparameterise(self.mu, self.std)
        return self.decoder(self.z)
        #return self.decoder(self.mu)



class CNN_encoder(nn.Module):
    def __init__(self, hidden, dr, channels, z_dim):
        super(CNN_encoder, self).__init__()
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

        self.z_dim  =\
                math.ceil(max(min(math.ceil(128*hidden),z_dim),math.ceil(64*hidden)))
        self.FC1  = nn.Linear(math.ceil(128*hidden), self.z_dim*2)

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

    def _init_weights(self):
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
        x = self.dropout(x)
        x = self.FC1(x)

        return x[:,self.z_dim:], F.softplus(x[:,self.z_dim:]-5, beta=1)

    def reparameterise(self, mean, std):
        eps = torch.randn_like(std)
        return mean + std*eps

    def get_mu_std(self):
        return self.mu, self.std

    def get_z_dim(self):
        return self.z_dim

    def forward(self, x):
        self.mu, self.std = self.encoder(x)
        return self.reparameterise(self.mu, self.std)




class decoder(nn.Module):
    def __init__(self,z_dim, hidden=1, outputs=2, dr=0.5):
        super(decoder, self).__init__()
        self.z_dim  =\
                math.ceil(max(min(math.ceil(128*hidden),z_dim),math.ceil(64*hidden)))

        self.FC1  = nn.Linear(self.z_dim,  math.ceil(64*hidden))
        self.FC2  = nn.Linear(math.ceil(64*hidden),  math.ceil(16*hidden))
        self.FC3  = nn.Linear(math.ceil(16*hidden),  math.ceil(8*hidden))
        self.FC4  = nn.Linear(math.ceil(8*hidden),  math.ceil(2*outputs))

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

    def forward(self, x):
        #x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.dropout(self.LeakyReLU(self.FC3(x)))
        x = self.FC4(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,2:4] = torch.square(x[:,2:4])
        return y[:,:2], y[:,2:]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)



class classifier(nn.Module):

    def __init__(self, z_dim, num_models=2):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(classifier,self).__init__()

        self.num_models     = num_models

        self.nn_classifier = nn.Sequential(nn.Linear(z_dim, 128),
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

