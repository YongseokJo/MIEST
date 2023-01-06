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
seed = 73
#torch.manual_seed(seed)
#np.random.seed(seed)

class VIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(VIB,self).__init__()
        
        
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        
        self.nn_encoder = nn.Sequential(
                            nn.Linear(self.input_shape,512),
                            nn.GELU(),
                            nn.LayerNorm(512),               
                            nn.Linear(512,128),
                            nn.LayerNorm(128),
                            #nn.Linear(256,256),
                            nn.ReLU(),
                            #nn.LayerNorm(256),
                            #nn.Linear(1024,128),
                            #nn.GELU(),
                            #nn.LayerNorm(128),
                            )

        #self.nn_token  = nn.Linear(128, self.coef_dim) 
        #self.nn_prob = nn.Linear(512, self.z_dim)

        self.nn_weights  = nn.Linear(128, self.z_shape) 
        self.nn_std   = nn.Linear(128, self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        #nn.Linear(64, 16),
                                        #nn.GELU(),
                                        #nn.LayerNorm(16),
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
        return self.nn_decoder(z)

    
    
    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)       
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps

    
    def forward(self, x):
        """
        Forward pass 
        
        Parameters:
        -----------
        x : (input_shape)
        """
        mu, std = self.encoder(x)
        z = self.reparameterise(mu, std)
        return self.decoder(z), mu, std
    
    
class linearVIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(linearVIB,self).__init__()
        
        
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        
        """
        self.nn_encoder = nn.Sequential(
                            nn.Linear(self.input_shape,512),
                            nn.ReLU(),
                            )
                            """

        self.nn_weights  = nn.Linear(self.input_shape, self.z_shape) 
        self.nn_std   = nn.Linear(self.input_shape, self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        #nn.Linear(64, 16),
                                        #nn.GELU(),
                                        #nn.LayerNorm(16),
                                        nn.Linear(32, self.output_shape))
        
    def encoder(self, x):
        """
        x : (input_shape)
        """
        #x = self.nn_encoder(x)
        #return self.nn_token(x), F.softplus(self.nn_prob(x)-5, beta=1)
        return self.nn_weights(x), F.softplus(self.nn_weights(x)-5, beta=1)

    def decoder(self, z):
        """
        z : (candidate_size)
        """ 
        return self.nn_decoder(z)

    
    
    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)       
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps

    
    def forward(self, x):
        """
        Forward pass 
        
        Parameters:
        -----------
        x : (input_shape)
        """
        mu, std = self.encoder(x)
        z = self.reparameterise(mu, std)
        return self.decoder(z), mu, std   
    
    

    
     
class polyVIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim, degree):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(polyVIB,self).__init__()
        
        
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        self.degree         = degree
        
        #print(self._polynomial_features(x))

        self.nn_weights  = nn.Linear(self.input_shape*len(self.degree), self.z_shape) 
        self.nn_std      = nn.Linear(self.input_shape*len(self.degree), self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 128),
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
        #print(x.size())
        x = self._polynomial_features(x)
        #print(x.size())
        return self.nn_weights(x), F.softplus(self.nn_weights(x)-5, beta=1)

    def decoder(self, z):
        """
        z : (candidate_size)
        """ 
        return self.nn_decoder(z)
    
    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)       
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps
    
    def _polynomial_features(self, x):
        #x = x.unsqueeze(1)
        #print(x.size())
        return torch.cat([x ** i for i in self.degree], 1)   

    
    def forward(self, x):
        """
        Forward pass 
        
        Parameters:
        -----------
        x : (input_shape)
        """
        mu, std = self.encoder(x)
        z = self.reparameterise(mu, std)
        return self.decoder(z), mu, std   
    
    
    
    
    
def vib_loss(y_pred, y, mu, std, beta=0.01):
    """    
    y_pred : (output_shape)
    y      : (output_shape)    
    mu     : (z_dim)  
    std    : (z_dim)
    """   
    #CE = F.cross_entropy(y_pred, y, reduction='sum')
    CE = torch.sum((y-y_pred)**2)
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return (beta*KL + CE) / y.size(0)



def train_vib(vib, train_loader, device,epochs=100,batch_size=50,
              test_dataset=None,learning_rate=1e-3,decay_rate=0.97):
    
    # Optimiser
    optimiser = torch.optim.Adam(vib.parameters(), lr=learning_rate)
    #optimiser = torch.optim.SGD(vib.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decay_rate)

    
    # Send to GPU if available
    vib.to(device)
    
    # Training
    measures = defaultdict(list)
    start_time = time.time()

    # put Deep VIB into train mode 
    vib.train()  

    for epoch in range(epochs):
        epoch_start_time = time.time()  

        # exponential decay of learning rate every 2 epochs
        if epoch % 10 == 0 and epoch > 0:
            scheduler.step()    
            pass

        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(train_loader): 
            X = X.to(device)        
            y = y.float().to(device)

            vib.zero_grad()
            y_pred, mu, std = vib(X)
            loss = vib_loss(y_pred, y, mu, std)
            loss.backward()
            optimiser.step()  

            batch_loss += loss.item()*X.size(0) 
            batch_accuracy += torch.sum(torch.abs(y - y_pred)/y)/batch_size     
            
        if test_dataset is not None:
            X, y = test_dataset.tensors
            y_pred, _,_ = vib(torch.tensor(X,dtype=torch.float).to(device))
            y_pred = y_pred.cpu().detach().numpy()
            test_accuray = np.abs(y-y_pred)/y
            #print(test_accuray.max())
            test_accuray = test_accuray.mean(axis=0)
            #test_accuray = test_accuray.mean()
            #print(test_accuray)
        else:
            test_accuray = -1.0
        # Save losses per epoch
        measures['total_loss'].append(batch_loss / len(train_loader.dataset))        
        # Save accuracy per epoch
        measures['accuracy'].append(batch_accuracy.cpu().detach() / len(train_loader.dataset))            
        if (epoch + 1) % 100 == 0 and epoch > 0:
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Loss: {:.4f}...".format(measures['total_loss'][-1]),
                  "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
                  "Test Om: {:.3f} sig: {:.3f}".format(test_accuray[0],test_accuray[1]))
                  #"Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))
            
    #torch.save(vib.state_dict(),"./models/model_LH_DMO_J_{}_L_{}_sigma_{}_power_{}.npy".format(J,L,sigma,integral_powers))
    return np.mean(measures['total_loss'][-10:]), np.mean(measures['accuracy'][-10:])




class newVIB(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(newVIB,self).__init__()
        
        
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        
        self.nn_encoder = nn.Sequential(
                            nn.Linear(self.input_shape,512),
                            nn.GELU(),
                            nn.LayerNorm(512),               
                            nn.Linear(512,128),
                            nn.LayerNorm(128),
                            #nn.Linear(256,256),
                            nn.ReLU(),
                            #nn.LayerNorm(256),
                            #nn.Linear(1024,128),
                            #nn.GELU(),
                            #nn.LayerNorm(128),
                            )

        #self.nn_token  = nn.Linear(128, self.coef_dim) 
        #self.nn_prob = nn.Linear(512, self.z_dim)

        self.nn_weights  = nn.Linear(128, self.z_shape) 
        self.nn_std   = nn.Linear(128, self.z_shape)

        self.nn_decoder = nn.Sequential(nn.Linear(self.z_shape, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        #nn.Linear(64, 16),
                                        #nn.GELU(),
                                        #nn.LayerNorm(16),
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
        return self.nn_decoder(z)

    
    
    def reparameterise(self, mean, std):
        """
        mean : (coef_dim)
        std  : (coef_dim)       
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mean + std*eps

    def get_latent_variable(self):
        return self.mu, self.std
    
    def forward(self, x):
        """
        Forward pass 
        
        Parameters:
        -----------
        x : (input_shape)
        """
        self.mu, self.std = self.encoder(x)
        z = self.reparameterise(self.mu, self.std)
        return self.decoder(z)
    
    
    
       

def new_train_vib(vib, train_loader, device,epochs=100,batch_size=50,
              test_dataset=None,learning_rate=1e-3,decay_rate=0.97):
    
    # Optimiser
    optimiser = torch.optim.Adam(vib.parameters(), lr=learning_rate)
    #optimiser = torch.optim.SGD(vib.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decay_rate)

    
    # Send to GPU if available
    vib.to(device)
    
    # Training
    measures = defaultdict(list)
    start_time = time.time()

    # put Deep VIB into train mode 
    vib.train()  

    for epoch in range(epochs):
        epoch_start_time = time.time()  

        # exponential decay of learning rate every 2 epochs
        if epoch % 10 == 0 and epoch > 0:
            scheduler.step()    
            pass

        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(train_loader): 
            X = X.to(device)        
            y = y.float().to(device)

            vib.zero_grad()
            y_pred = vib(X)
            mu, std = vib.get_latent_variable()
            loss = vib_loss(y_pred, y, mu, std)
            loss.backward()
            optimiser.step()  

            batch_loss += loss.item()*X.size(0) 
            batch_accuracy += torch.sum(torch.abs(y - y_pred)/y)/batch_size     
            
        if test_dataset is not None:
            X, y = test_dataset.tensors
            y_pred = vib(torch.tensor(X,dtype=torch.float).to(device))
            y_pred = y_pred.cpu().detach().numpy()
            test_accuray = np.abs(y-y_pred)/y
            #print(test_accuray.max())
            test_accuray = test_accuray.mean(axis=0)
            #test_accuray = test_accuray.mean()
            #print(test_accuray)
        else:
            test_accuray = -1.0
        # Save losses per epoch
        measures['total_loss'].append(batch_loss / len(train_loader.dataset))        
        # Save accuracy per epoch
        measures['accuracy'].append(batch_accuracy.cpu().detach() / len(train_loader.dataset))            
        if (epoch + 1) % 100 == 0 and epoch > 0:
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Loss: {:.4f}...".format(measures['total_loss'][-1]),
                  "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
                  "Test Om: {:.3f} sig: {:.3f}".format(test_accuray[0],test_accuray[1]))
                  #"Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))
            
    #torch.save(vib.state_dict(),"./models/model_LH_DMO_J_{}_L_{}_sigma_{}_power_{}.npy".format(J,L,sigma,integral_powers))
    return np.mean(measures['total_loss'][-10:]), np.mean(measures['accuracy'][-10:])





class NDR(nn.Module):
    """An implementation of the Variational Information Bottleneck Method."""

    def __init__(self, input_shape, output_shape, z_dim, num_models=2):
        # We'll use the same encoder as before but predict additional parameters
        #  for our distribution.
        super(NDR,self).__init__()
        
        
        self.input_shape    = input_shape
        self.output_shape   = output_shape
        self.z_shape        = z_dim #self.coef_dim*(self.coef_dim+3)/2
        self.num_models     = num_models
        
        self.nn_encoder = nn.Sequential(
            nn.Linear(self.input_shape,1024),
            nn.GELU(),
            nn.LayerNorm(1024),               
            nn.Linear(1024,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.z_shape) 
        )
        
        self.nn_classifier = nn.Sequential(nn.Linear(self.z_shape, 128),
                                        nn.GELU(),
                                        nn.LayerNorm(128),
                                        nn.Linear(128, 32),
                                        nn.GELU(),
                                        nn.LayerNorm(32),
                                        nn.Linear(32, self.num_models))
        
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
        return self.nn_encoder(x)
        #return self.nn_token(x), F.softplus(self.nn_prob(x)-5, beta=1)
        #return self.nn_weights(x), F.softplus(self.nn_weights(x)-5, beta=1)

    def classifier(self,z):
        return self.nn_classifier(z)
        
        
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
    
    def get_class(self):
        return self._class
    
    def forward(self, x):
        """
        Forward pass 
        
        Parameters:
        -----------
        x : (input_shape)
        """
        self.z = self.encoder(x)
        self._class = self.classifier(self.z)
        #print(self._class)
        #z = self.reparameterise(self.mu, self.std)
        return self.decoder(self.z)
    
 
    
def ndr_loss(y_mean,y_sigma, y, y_class, _class, beta=0.01):
    """    
    y_pred : (output_shape)
    y      : (output_shape)    
    mu     : (z_dim)  
    """   
    alpha=1.
    J0 = torch.sum((y-y_mean)**2).mean()
    J1 = torch.sum(((y-y_mean)**2-y_sigma**2)**2).mean()
    #print(L2)
    CE = nn.CrossEntropyLoss()
    ce=CE(y_class, _class)
    #print(ce)
    #return torch.log(J0)+alpha*torch.log(J1)-beta*ce
    #return torch.log(J0+J1)-beta*ce
    return J0+J1-beta*ce



def train_ndr(vib, train_loader, device,epochs=100,batch_size=50,
              test_dataset=None,learning_rate=1e-3,decay_rate=0.97,beta=1e-2,
             verbose=True):
    
    # Optimiser
    optimiser = torch.optim.Adam(vib.parameters(), lr=learning_rate)
    #optimiser = torch.optim.SGD(vib.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decay_rate)

    
    # Send to GPU if available
    vib.to(device)
    
    # Training
    measures = defaultdict(list)
    start_time = time.time()

    # put Deep VIB into train mode 
    vib.train()  

    for epoch in range(epochs):
        epoch_start_time = time.time()  

        # exponential decay of learning rate every 2 epochs
        if epoch % 10 == 0 and epoch > 0:
            scheduler.step()    
            pass

        batch_loss = 0
        batch_accuracy = 0
        for _, (X,y) in enumerate(train_loader): 
            X = X.to(device)        
            y = y.float().to(device)
            y_class = y[:,2:]
            y = y[:,:2]

            vib.zero_grad()
            y_pred,y_sigma = vib(X)
            _mu = vib.get_latent_variable()
            _class =  vib.get_class()
            loss = ndr_loss(y_pred, y_sigma, y, y_class, _class, beta=beta)
            loss.backward()
            optimiser.step()  

            batch_loss += loss.item()*X.size(0) 
            batch_accuracy += torch.sum(torch.abs(y - y_pred)/y)/batch_size     
            
        if test_dataset is not None:
            X, y = test_dataset.tensors
            y_pred, y_sigma = vib(torch.tensor(X,dtype=torch.float).to(device))
            y_pred = y_pred.cpu().detach().numpy()
            test_accuray = np.abs(y-y_pred)/y
            #print(test_accuray.max())
            test_accuray = test_accuray.mean(axis=0)
            #test_accuray = test_accuray.mean()
            #print(test_accuray)
        else:
            test_accuray = -1.0
        # Save losses per epoch
        measures['total_loss'].append(batch_loss / len(train_loader.dataset))        
        # Save accuracy per epoch
        measures['accuracy'].append(batch_accuracy.cpu().detach() / len(train_loader.dataset))            
        if (epoch + 1) % 100 == 0 and epoch > 0 and verbose:
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Loss: {:.4f}...".format(measures['total_loss'][-1]),
                  "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
                  "Test Om: {:.3f} sig: {:.3f}".format(test_accuray[0],test_accuray[1]))
                  #"Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))
            
    #torch.save(vib.state_dict(),"./models/model_LH_DMO_J_{}_L_{}_sigma_{}_power_{}.npy".format(J,L,sigma,integral_powers))
    return np.mean(measures['total_loss'][-10:]), np.mean(measures['accuracy'][-10:])

