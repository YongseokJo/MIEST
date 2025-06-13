import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
import sklearn
import sys,os
from functools import partial
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from vib_utils import *
from networks import *
import warnings
warnings.filterwarnings('ignore')
#import xgboost as xgb
from copy import deepcopy
import math



class MIST():
    mach_vib_cls = ['vib+cls', 'vib+cls_a']
    is_cls = ['vib+cls', 'vib+cls_a', 'vib_cnn+cls']
    mach_vib      = ['vib', 'vib_cnn']
    mach_no_vib   = ['fcl', 'cnn']
    
    def __init__(self, sim="TNG", field="Mtot", batch_size=100, 
                 L=None, dn=None, data_type='wph', proc_imag=None,
                 normalization=False, monopole=True, projection=True,
                 use_only_om=False, use_only_sig=False, guide_output=False,
                 average=True, device='cuda', robust_axis='sim'):
        self.sims          = sim
        self.field         = field
        self.batch_size    = batch_size
        self.normalization = normalization
        self.projection    = projection
        self.monopole      = monopole
        self.average       = average
        self.device        = device
        self.num_sim       = len(self.sims) if isinstance(self.sims, list) else 1
        self.L             = L
        self.dn            = dn
        self.use_only_om   = use_only_om
        self.use_only_sig  = use_only_sig
        self.guide_output  = guide_output
        self.proc_imag     = proc_imag
        self.data_type     = data_type
        self.robust_axis   = robust_axis

        if self.robust_axis == 'sim':
            self.N_robust = self.num_sim
        elif self.robust_axis == 'field':
            self.N_robust = len(self.field)

        if data_type == 'image':
            self.projection = False
            self.average    = False
        elif projection:
            if not monopole:
                print("You can't use no monopole with projection!")
                print("Turning on monopole...")
                self.monopole = True
            if average:
                print("You can't use average with projection!")
                print("Turning off average...")
                self.average = False

        if data_type == 'pywst':
            self.projection = False
            self.average    = False

        self.input, self.output = self.load_data()

        self.make_train_set()



    def train(self,which_machine="vib+cls",fname="test",beta=1, gamma=1e-2, 
              learning_rate=1e-3, max_lr=1e-9, wd=1e-2,
              z_dim=200, epochs=3000, hidden1=0.5, hidden2=0.25, dr=0.5,channels=1,
              patience=50, tol=1e-3,
              verbose=True, save_plot=True, save_model=True,
              n_estimators=10, max_depth=4, loss_fn=None):

        self.beta = beta; self.gamma = gamma;
        self.which_machine = which_machine
        if self.robust_axis == 'sim':
            num_models = self.num_sim
        elif self.robust_axis == 'field':
            num_models = len(self.field)
        if which_machine == 'xgboost':
            """
            X_train, y_train = self.train_loader.dataset.tensors
            X_train, y_train = X_train[self.train_indices,:], y_train[self.train_indices,:2]
            # create model instance
            self.bst =xgb.XGBRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth, learning_rate=1,
                                       objective='reg:squaredlogerror')
            # fit model
            self.bst.fit(X_train, y_train)
            success = True
            """
        elif which_machine == 'tabnet':
            self.rgs = TabNetRegressor()  #TabNetRegressor()
            aug = RegressionSMOTE(p=0.2)
            X, y = np.array(self.train_loader.dataset.tensors)
            X_train, y_train = np.array(X[self.train_indices,:]),\
                    np.array(y[self.train_indices,:2])
            X_valid, y_valid = np.array(X[self.val_indices,:]),\
                    np.array(y[self.val_indices,:2])

            if loss_fn is None:
                loss_fn = nn.MSELoss()


            self.rgs.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_name=['train', 'valid'],
                eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
                max_epochs=epochs,
                patience=50,
                batch_size=self.batch_size,
                virtual_batch_size=int(self.batch_size/10),
                num_workers=0,
                drop_last=False,
                augmentations=aug,
                loss_fn=loss_fn,
            )
            success =True
        else:
            if which_machine == 'fcl':
                self.vib = FCL(self.input.shape[1], self.output.shape[1])
                self.cls = None
            elif which_machine == 'cnn':
                self.vib = CNN(hidden=hidden1, dr=dr,channels=channels)
                self.cls = None
            elif which_machine == 'vib':
                self.vib = VIB(self.input.shape[1], self.output.shape[1],
                               z_dim, fe=hidden1, fd=hidden2, dr=dr)
                self.cls = None
            elif which_machine == 'vib_cnn':
                self.vib = VIB_CNN(hidden=hidden1, dr=dr,channels=channels,
                                   z_dim=z_dim)
                self.cls = None
            elif which_machine == 'vib_cnn+cls':
                self.vib = VIB_CNN(hidden=hidden1, dr=dr,channels=channels,
                                   z_dim=z_dim)
                self.cls = classifier(self.vib.get_z_dim(),
                                      num_models=num_models)
            elif which_machine == 'cnn_enc_dec':
                self.vib = cnn_encoder_decoder(hidden=hidden1, dr=dr,channels=channels,
                                   z_dim=z_dim)
                self.cls = classifier(self.vib.get_z_dim(),
                                      num_models=num_models)
            elif which_machine == "vib+cls":
                self.vib = VIB(self.input.shape[1], self.output.shape[1],
                               z_dim, fe=hidden1, fd=hidden2, dr=dr)
                self.cls = classifier(z_dim, num_models=num_models)
            elif which_machine == "separate_test":
                self.encoder = CNN_encoder(hidden=hidden1, dr=dr,channels=channels,
                                           z_dim=z_dim)
                self.decoder = decoder(z_dim, hidden=hidden1, dr=dr)
                self.cls = classifier(self.encoder.get_z_dim(), num_models=self.num_sim)

                self.trainer = Trainer(None, None, self.train_loader, which_machine,
                                       self.valid_loader,fname,self.device, num_sim=self.num_sim)
                res     = self.trainer.run_cnn_separate(
                    self.encoder, self.decoder, self.cls,
                    base_lr=learning_rate,max_lr=max_lr,
                    epochs=epochs, beta=beta, gamma=gamma,
                    patience=patience, tol=tol, wd=wd,# beta1, beta2,
                    verbose=verbose, save_plot=save_plot, save_model=save_model)
                return

            self.trainer = Trainer(self.vib, self.cls, self.train_loader, which_machine,
                              self.valid_loader,fname,self.device,
                                   num_sim=num_models)

            if which_machine in ['vib_cnn', 'cnn', 'vib_cnn+cls', 'cnn_enc_dec']:
                __train__ = self.trainer.run_cnn
            else:
                __train__ = self.trainer.run
            res     = __train__(base_lr=learning_rate,max_lr=max_lr,
                                epochs=epochs, beta=beta, gamma=gamma,
                                patience=patience, tol=tol, wd=wd,# beta1, beta2,
                                verbose=verbose, save_plot=save_plot, save_model=save_model)
            del self.trainer
            self.vib, self.cls, success = res
        return success



    def retrain(self, learning_rate=1e-3,epochs=3000,patience=50, tol=1e-3,
                verbose=True, save_plot=True, save_model=True,):
        res = self.trainer.run(learning_rate, epochs, 
                               self.beta, self.gamma, patience, tol,
                               verbose, save_plot=save_plot, save_model=save_model)
        self.vib, self.cls, success = res
        return success


    def get_valid_loss(self):
        self.vib.eval()
        val_loss = 0.
        for N, (X_val, y_val) in enumerate(self.valid_loader):
            with torch.no_grad():
                y_param        = y_val[:,:2].cpu().detach().numpy()
                y_pred, _      = self.vib(X_val.to(self.device))
                y_pred         = y_pred.detach().cpu().numpy()
                diff           = np.mean((y_param-y_pred)**2,axis=0)
                val_loss      += np.sum(np.log(diff))
        return val_loss/(N+1)

    def get_auc_score(self):
        X,y   = self.val_dataset.tensors
        y_cls = y[:,-self.N_robust:].cpu().detach().numpy()
        y_pred, y_std, pred_cls = self.predict_cnn(X, cls=True)
        auc   = sklearn.metrics.roc_auc_score(y_cls, pred_cls)
        return auc #np.abs(auc/(N+1)-0.5)#**2*1e-2

    def retrain_cls(self, cls=None):
        if cls == None:
            cls = self.cls
        self.trainer = Trainer(None, self.cls, self.train_loader, "cls",
                               self.valid_loader,None,self.device,
                               num_sim=self.N_robust)


    def make_train_set(self,imaginary=False):
        train_split     = .8
        val_split       = .1
        shuffle_dataset = True
        random_seed     = 42

        # Labelling for simultions

        N_sample = self.output.shape[0]
        N_output = self.output.shape[1] 
        y        = torch.zeros((N_sample, 
                                N_output + self.N_robust))
        y[:,:N_output] = torch.tensor(self.output, dtype=torch.float)
        if self.projection:
            N = 3000
        elif self.average:
            N = 1000
        else:
            N = 15000
        for i in range(self.N_robust):
            y[i*N:(i+1)*N, N_output + i] = 1.
        self.output_cls = y[:,-self.N_robust:]

        X = torch.tensor(self.input,dtype=torch.float)
        dataset      = data_utils.TensorDataset(X, y)
        if self.projection:
            denom = 3
        elif self.average:
            denom = 1
        else:
            denom = 15
        dataset_size = int(len(dataset)/denom)
        indices      = np.array(list(range(dataset_size)))
        train_split  = int(np.floor(train_split * dataset_size))
        val_split    = int(np.floor(val_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        self.train_indices = []
        for at in indices[:train_split]:
            self.train_indices += list(range(int(15*at),int(15*(at+1))))
        self.train_indices = np.array(self.train_indices)

        self.val_indices = []
        for at in indices[train_split:train_split+val_split]:
            self.val_indices += list(range(int(15*at),int(15*(at+1))))
        self.val_indices = np.array(self.val_indices)

        self.test_indices = []
        for at in indices[train_split+val_split:]:
            self.test_indices += list(range(int(15*at),int(15*(at+1))))
        self.test_indices = np.array(self.test_indices)

        """
        self.train_indices = indices[:train_split]
        self.val_indices   = indices[train_split:train_split+val_split]
        self.test_indices  = indices[train_split+val_split:]
        """


        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(self.train_indices)
        valid_sampler = SubsetRandomSampler(self.val_indices)
        test_sampler  = SubsetRandomSampler(self.test_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                        sampler=train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                        sampler=valid_sampler)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                        sampler=test_sampler)
        self.test_dataset = data_utils.TensorDataset(X[self.test_indices],y[self.test_indices])
        self.val_dataset  = data_utils.TensorDataset(X[self.val_indices], y[self.val_indices])


    def save_models(self,fname="test",fpath=None):
        if fpath is None:
            fpath = "./model"
        torch.save(self.vib.state_dict(),"{}/{}_vib.pt".format(fpath, fname))
        if self.which_machine in self.is_cls:
            torch.save(self.cls.state_dict(),"{}/{}_cls.pt".format(fpath, fname))


    def load_models(self,fname="test", which_machine="vib+cls", 
                    z_dim=200, hidden1=1, hidden2=0.25, dr=0.5, channels=1,
                    fpath=None):
        if fpath is None:
            fpath = './model'
        self.which_machine = which_machine
        ## I have to add multiple inputs
        if which_machine == 'fcl':
            self.vib = FCL(self.input.shape[1], self.output.shape[1])
        elif which_machine == 'cnn':
            self.vib = CNN(hidden=hidden1, dr=dr,channels=channels)
        elif which_machine == 'vib_cnn':
            self.vib = VIB_CNN(hidden=hidden1,
                               dr=dr,channels=channels,z_dim=z_dim)
        elif which_machine == 'vib_cnn+cls':
            self.vib = VIB_CNN(hidden=hidden1,
                               dr=dr,channels=channels,z_dim=z_dim)
        elif which_machine == 'vib':
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim, fd=hidden2, fe=hidden1,dr=dr)
        elif which_machine == 'vib+cls':
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim, fd=hidden2, fe=hidden1,dr=dr)
        elif which_machine == 'vib+cls_a':
            self.vib = VIB_a(self.input.shape[1], self.output.shape[1],
                           z_dim=z_dim, h=hidden, dr=dr)

        self.vib.load_state_dict(torch.load("{}/{}_vib.pt".format(fpath, fname)))
        self.vib.to(self.device).eval()
        if which_machine in self.mach_vib_cls:
            self.cls = classifier(z_dim, num_models=self.N_robust)
            self.cls.load_state_dict(torch.load("{}/{}_cls.pt".format(fpath, fname)))
            self.cls.to(self.device).eval()


    def load_optuna_models(self, storage, study_name, metric=None, suffix='',
                           which_machine="vib_cnn", num_trial=None,
                           print_loss=True, channels=1, fpath=None, z_dim=None):
        self.which_machine = which_machine
        import optuna_utils as op
        if fpath == None:
            fpath = './model/optuna/'
        if fpath is not None:
            fpath = f'{fpath}/model/optuna/'
        num_trial, params, self.trial = op.best_params(study_name, storage,
                                           verbose=False,
                                           metric=metric,
                                           num_trial=num_trial)
        if self.which_machine == 'vib':
            fname = '{}_{}'.format(study_name, num_trial)
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'], fe=params['fe'],
                           fd=params['fd'], dr=params['dropout'])
            self.cls = None
        elif which_machine == 'vib_cnn':
            fname = '{}_{}'.format(study_name, num_trial)
            self.vib = VIB_CNN(hidden=params['hidden'], dr=params['dropout'],
                               channels=channels,
                               z_dim=params['z_dim'])#params['z_dim'])
            self.cls = None
        elif which_machine == 'vib_cnn+cls':
            fname = '{}_{}'.format(study_name, num_trial)
            self.vib = VIB_CNN(hidden=params['hidden'], dr=params['dropout'],
                               channels=channels,
                               z_dim=params['z_dim'])#params['z_dim'])
            self.cls = classifier(z_dim=self.vib.get_z_dim(), num_models=self.N_robust)
        elif which_machine == 'cnn_enc_dec':
            fname = '{}_{}'.format(study_name, num_trial)
            if z_dim is None:
                z_dim = params['z_dim']
            self.vib = cnn_encoder_decoder(hidden=params['hidden'], dr=params['dropout'],
                               channels=channels,
                               z_dim=z_dim)#params['z_dim'])
            self.cls = classifier(z_dim=self.vib.get_z_dim(), num_models=self.N_robust)
        elif which_machine == 'vib+cls':
            fname = '{}_{}'.format(study_name, num_trial)
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'] , fe=params['fe'],
                           fd=params['fd'], dr=params['dropout'])
            self.cls = classifier(z_dim=params['z_dim'], num_models=self.N_robust)
        elif which_machine == 'vib+cls_a' or which_machine == 'vib+cls_a':
            fname = '{}_{}_a'.format(study_name, num_trial)
            self.vib = VIB_a(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'],h=params['hidden'],
                           dr=params['dropout'])
            self.cls = classifier(z_dim=params['z_dim'], num_models=self.N_robust)

        self.vib.load_state_dict(torch.load("{}/{}_{}vib.pt".format(fpath,fname,suffix),
                                           map_location=self.device))
        self.vib.to(self.device).eval()
        if self.which_machine in self.is_cls:
            self.cls.load_state_dict(torch.load("{}/{}_{}cls.pt".format(fpath,fname,suffix),
                                               map_location=self.device))
            self.cls.to(self.device).eval()
        if print_loss and which_machine in self.mach_vib:
            print("num_trial={}, params={}, values={} ".format(num_trial,
                                                               params,
                                                               self.trial.values))
            pass
            #print("gamma={}".format(params['gamma']))
        if print_loss and which_machine in self.mach_vib_cls:
            print("fe={}, fd={}, z dim={}".format(params['fe'],params['fd'],params['z_dim']))
            pass
            #print("beta={}, gamma={}".format(params['beta'], params['gamma']))


    def load_data(self, external=False, external_sims=None,):
        sim_names = {"TNG"   :"IllustrisTNG",
                     "SIMBA" :"SIMBA",
                     "ASTRID":"Astrid",
                     "EAGLE":"EAGLE",
                     "GADGET": "Gadget",
                     "RAMSES": "Ramses",
                     "TNG_SB":"IllustrisTNG",
                     "MAGNETICUM":"Magneticum",
                    }
        fparams =\
        {"TNG"   :'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/params_LH_IllustrisTNG.txt',
         "SIMBA" :'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/SIMBA/params_LH_SIMBA.txt',
         "ASTRID":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Astrid/params_LH_Astrid.txt',
         "TNG_SB":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/params_SB28_IllustrisTNG.txt',
         "EAGLE":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/EAGLE/params_LH_EAGLE.txt',
         "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/params_Ramses.txt",
         "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/params_Gadget.txt",
         #"TNG_SB":"/mnt/home/fvillaescusa/CAMELS/Results/images_IllustrisTNG_SB28/params_IllustrisTNG_SB28_6.txt",
         "MAGNETICUM":"/mnt/home/fvillaescusa/CAMELS/Results/images_Magneticum/params_Magneticum.txt.txt",
                  }
        if external:
            if external_sims is None:
                print("You need to pass simulation type!")
            else:
                sims = external_sims
        else:
            sims  = self.sims

        if self.data_type == 'image':
            fmaps = \
                    {'TNG'   :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_LH_z=0.00.npy".format(self.field),
                     'SIMBA' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/SIMBA/Maps_{}_SIMBA_LH_z=0.00.npy".format(self.field),
                     'ASTRID':"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Astrid/Maps_{}_Astrid_LH_z=0.00.npy".format(self.field),
                     "TNG_SB":"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_SB28_z=0.00.npy".format(self.field),
                     'EAGLE' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/EAGLE/Maps_{}_EAGLE_LH_z=0.00.npy".format(self.field),
                     "MAGNETICUM":"/mnt/home/fvillaescusa/CAMELS/Results/images_Magneticum/Images_{}_Magneticum_LH_0_z=0.00.npy".format(self.field),
                     "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/Images_M_Ramses_LH_z=0.00.npy",
                     "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/Images_M_Gadget_LH_z=0.00.npy",
                     "AREPO_SIMBA":"/mnt/ceph/users/fgarcia/data_products/simba_test_latest_temp_maps_z0/gas_temperature_033.npy",
                    }

            if self.robust_axis == 'sim':
                if isinstance(sims, list):
                    maps = []
                    params    = []
                    for sim in sims:
                        map_ = self.__img_loader(sim, self.field, sim_names)
                        #map_ = np.log10(np.load(fmaps[sim]))
                        #map_ = map_.reshape(-1,1,map_.shape[1],map_.shape[2])
                        maps.append(map_)
                        param     = np.loadtxt(fparams[sim])[:,:2]
                        n_sample  = param.shape[0]
                        param_ext = np.ones((int(n_sample*15),2))
                        for i in range(n_sample):
                            param_ext[i*15:(i+1)*15,:] = \
                                    param[i,:]
                        params.append(param_ext)  ## only Om and Sig8
                    _input  = np.vstack(maps)
                    _output = np.vstack(params)
                else:
                    _input = self.__img_loader(sims,self.field,sim_names)
                    param     = np.loadtxt(fparams[sims])[:,:2]
                    n_sample  = param.shape[0]
                    _output = np.ones((int(15*n_sample),2))
                    for i in range(n_sample):
                        _output[i*15:(i+1)*15,:] = \
                                    param[i,:]
            elif self.robust_axis == 'field':
                maps   = []
                params = []
                for field in self.field:
                    sim = sims
                    map_ = self.__img_loader(sim, field, sim_names)
                    maps.append(map_)
                    param     = np.loadtxt(fparams[sim])[:,:2]
                    n_sample  = param.shape[0]
                    param_ext = np.ones((int(n_sample*15),2))
                    for i in range(n_sample):
                        param_ext[i*15:(i+1)*15,:] = \
                                param[i,:]
                    params.append(param_ext)  ## only Om and Sig8
                _input  = np.vstack(maps)
                _output = np.vstack(params)



        elif self.data_type == 'wph' or self.data_type == 'pywst':
            if self.L is None:
                suffix = ''
            else:
                suffix = "_l_{}_dn_{}".format(self.L, self.dn)

            if self.data_type == 'wph':
                loader = self.__wph_loader

            if self.data_type == 'pywst':
                loader = self.__wst_loader

            prefix = ''
            if not self.monopole:
                prefix = 'n'
            if self.average:
                N_sample = 1
            else:
                N_sample = 15
            if self.projection:
                prefix = 'p'
                N_sample = 3

            if isinstance(sims, list):
                coefs  = []
                params = []
                for sim in sims:
                    coef = loader(prefix, sim, self.field, suffix,
                                             sim_names)
                    """
                    if self.monopole:
                        mono = self.__monopole_loader(sim, self.field,sim_names)
                        mono = mono.reshape(-1,1)
                        mono = (mono-mono.min(axis=0))/(mono.max(axis=0)-mono.min(axis=0))
                        coef = np.hstack([coef,mono.reshape(-1,1)])
                        """

                    if self.average:
                        N_sample = 1
                        coef_avg = np.zeros((1000, coef.shape[1]))
                        for i in range(1000):
                            coef_avg[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                        coefs.append(coef_avg)
                    else:
                        coefs.append(coef)
                    param     = np.loadtxt(fparams[sim])[:,:2]
                    param_ext = np.ones((N_sample*1000,2))
                    for i in range(1000):
                        param_ext[i*N_sample:(i+1)*N_sample,:] = \
                                param[i,:]
                    params.append(param_ext)  ## only Om and Sig8

                _input  = np.vstack(coefs)
                _output = np.vstack(params)
            else:
                coef = loader(prefix, sims, self.field, suffix,
                                         sim_names)
                """
                if self.monopole:
                    mono = self.__monopole_loader(sims, self.field,sim_names)
                    mono = mono.reshape(-1,1)
                    mono = (mono-mono.min(axis=0))/(mono.max(axis=0)-mono.min(axis=0))
                    coef = np.hstack([coef,mono.reshape(-1,1)])
                    """

                _output = np.loadtxt(fparams[sims])[:,:2]

                if self.average:
                    N_sample = 1
                    _input = np.zeros((1000, coef.shape[1]))
                    for i in range(1000):
                        _input[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                else:
                    _input  = coef

                param   = deepcopy(_output)
                _output = np.ones((N_sample*1000,2))
                for i in range(1000):
                    _output[i*N_sample:(i+1)*N_sample,:] = \
                            param[i,:]


            if self.proc_imag is None:
                self.process_imaginary = self.__process_imag_to_phase
            else:
                self.process_imaginary  = self.proc_imag
            _input  = self.process_imaginary(deepcopy(_input))



        if self.normalization:
            if external:
                _input = (_input -self.mean_norm)/self.std_norm
            else:
                self.mean_norm = _input.mean()
                self.std_norm  = _input.std()
                _input = (_input -self.mean_norm)/self.std_norm

        """
        if self.use_only_om:
            _output[:,1]= 1
        if self.use_only_sig:
            _output[:,0]= 1
        if self.use_only_om and self.use_only_sig:
            raise
        if self.guide_output:
            _output_new = np.ones((_output.shape[0],_output.shape[1]+1))
            _output_new[:,:_output.shape[1]] = _output
            _output = _output_new
            """

        return _input, _output

    def __img_loader(self, sim, fields, sim_names):

        if isinstance(fields, list):
            img = []
            for field in fields:
                fmaps = \
                        {'TNG'   :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_LH_z=0.00.npy".format(field),
                         'SIMBA' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/SIMBA/Maps_{}_SIMBA_LH_z=0.00.npy".format(field),
                         'ASTRID':"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Astrid/Maps_{}_Astrid_LH_z=0.00.npy".format(field),
                         'EAGLE' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/EAGLE/Maps_{}_EAGLE_LH_z=0.00.npy".format(self.field),
                         "TNG_SB":"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_SB28_z=0.00.npy".format(self.field),
                     "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/Images_M_Ramses_LH_z=0.00.npy",
                     "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/Images_M_Gadget_LH_z=0.00.npy",
                     "AREPO_SIMBA":"/mnt/ceph/users/fgarcia/data_products/simba_test_latest_temp_maps_z0/gas_temperature_033.npy",
                        }
                img_tmp = np.load(fmaps[sim])
                img_tmp = np.log10(img_tmp).reshape(-1,1,256,256)
                if not self.monopole:
                    nmaps  = img_tmp.reshape(img_tmp.shape[0],-1)
                    nmaps  = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
                    nmaps  = nmaps.T.reshape(-1,1,img_tmp.shape[2],img_tmp.shape[3])
                    img_tmp = nmaps
                    del nmaps
                img.append(img_tmp)
            img = np.concatenate(img, axis=1)
        else:
            field = fields
            fmaps = \
                    {'TNG'   :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_LH_z=0.00.npy".format(field),
                     'SIMBA' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/SIMBA/Maps_{}_SIMBA_LH_z=0.00.npy".format(field),
                     'ASTRID':"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Astrid/Maps_{}_Astrid_LH_z=0.00.npy".format(field),
                     'EAGLE' :"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/EAGLE/Maps_{}_EAGLE_LH_z=0.00.npy".format(self.field),
                     "TNG_SB":"/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/IllustrisTNG/Maps_{}_IllustrisTNG_SB28_z=0.00.npy".format(self.field),
                     "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/Images_M_Ramses_LH_z=0.00.npy",
                     "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/Images_M_Gadget_LH_z=0.00.npy",
                     "AREPO_SIMBA":"/mnt/ceph/users/fgarcia/data_products/simba_test_latest_temp_maps_z0/gas_temperature_033.npy",
                    }
            img = np.load(fmaps[sim])
            img = np.log10(img).reshape(-1,1,256,256)
            if not self.monopole:
                nmaps  = img.reshape(img.shape[0],-1)
                nmaps  = (nmaps.T - nmaps.mean(axis=1))/nmaps.std(axis=1)
                nmaps  = nmaps.T.reshape(-1,1,img.shape[2],img.shape[3])
                img = nmaps
                del nmaps
        return img

    def __wph_loader(self, prefix, sim, field, suffix, sim_names):
        if isinstance(self.field, list):
            wph = []
            for field in self.field:
                wph.append(np.load(
                    "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}{}_{}_for_vib_total{}.npy"\
                    .format(prefix,sim_names[sim], field,
                            suffix)))
            wph = np.hstack(wph)
        else:
            wph = np.load(
                "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}{}_{}_for_vib_total{}.npy"\
                .format(prefix,sim_names[sim], self.field, suffix))
        return wph

    def __wst_loader(self, prefix, sim, field, suffix, sim_names):
        if isinstance(self.field, list):
            wph = []
            for field in self.field:
                wph.append(np.load(
                    "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/pywst_{}_{}_for_vib_total_.npy"\
                    .format(sim_names[sim], field)).reshape(15000,-1))
            wph = np.hstack(wph)
        else:
            wph = np.load(
                "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/pywst_{}_{}_for_vib_total_.npy"\
                .format(sim_names[sim], self.field)).reshape(15000,-1)
        return wph

    def __monopole_loader(self, sim, field, sim_names):
        if isinstance(self.field, list):
            monopole = []
            for field in self.field:
                monopole.append(np.load(
                    "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}_{}_monopole.npy"\
                    .format(sim_names[sim], field)))
            monopole = np.hstack(monopole)
        else:
            monopole = np.load(
                "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}_{}_monopole.npy"\
                .format(sim_names[sim], field))
        return np.log10(monopole)


    def get_latent_variable(self, X=None):
        if X is None:
            X = self.test_dataset.tensors[0]
        _, _, z = self.predict_cnn(X, latent=True)
        return z

    def get_mu(self, X=None):
        if X is None:
            X = self.test_dataset.tensors[0]
        _, _, z = self.predict_cnn(X, latent=False, mu=True)
        return z


    def predict_cnn(self,X, cls=False, latent=False, mu=False):
        y_pred = np.empty((X.shape[0],self.output.shape[1]),dtype=float)
        y_std  = np.empty((X.shape[0],self.output.shape[1]),dtype=float)
        z      = np.empty((X.shape[0],self.vib.get_z_dim()),dtype=float)
        sample_size = X.shape[0]

        if cls:
            y_cls  = np.empty((X.shape[0],self.N_robust),dtype=float)

        with torch.no_grad():
            for j in range(math.ceil(sample_size/self.batch_size)):
                start = j*self.batch_size
                end   = min((j+1)*self.batch_size, sample_size)
                if ~torch.is_tensor(X):
                    X_ = torch.tensor(X[start:end,:], dtype=torch.float).to(self.device)
                pred,    std        = self.vib(X_)
                y_pred[start:end,:] = pred.cpu().detach().numpy()
                y_std[start:end,:]  = std.cpu().detach().numpy()
                if latent:
                    z_              = self.vib.get_latent_variable()
                    z[start:end,:]  = z_.cpu().detach().numpy()
                    del z_
                if mu:
                    z_, _           = self.vib.get_mu_std()
                    z[start:end,:]  = z_.cpu().detach().numpy()
                    del z_
                del pred, std, X_
                if cls:
                    pred_cls           = self.cls(self.vib.get_latent_variable())
                    y_cls[start:end,:] = pred_cls.cpu().detach().numpy()
                    del pred_cls
                torch.cuda.empty_cache()

        res = [y_pred, y_std]
        if cls:
            res += [y_cls]
        if latent:
            res += [z]
        if mu:
            res += [z]
        return res


    def make_plots_cnn(self, fname="test", dpi=100, figsize=(20,20),
                   batch_size=None,
                   show_plot=True, data_return=False, save_plot=True):
        self.vib.eval()
        if self.which_machine in self.is_cls:
            self.cls.eval()

        ## Figure settings
        fontsize = 30
        plt.rcParams['font.size'] = '30'
        plt.rcParams['font.family'] = 'sans-serif'
        #plt.rcParams['font.sans-serif'] = 'sans-serif'
        plt.rcParams['xtick.labelsize'] = '15'
        plt.rcParams['ytick.labelsize'] = '14'

        fig = plt.figure(figsize=figsize)
        ideal1 = np.linspace(0.1,0.5,3)
        ideal2 = np.linspace(0.6,1.0,3)



        y_true = []
        y_pred = []
        y_std  = []

        for i in range(self.N_robust):
            index                 = np.logical_and(self.test_indices<(i+1)*15000,
                                                   self.test_indices>=i*15000)
            X                     = self.input[self.test_indices[index]]
            y_true_tmp            = self.output[self.test_indices[index]]
            y_pred_tmp, y_std_tmp = self.predict_cnn(X)

            y_true.append(np.array(y_true_tmp))
            y_pred.append(np.array(y_pred_tmp))
            y_std.append(np.array(y_std_tmp))

            #title = self.field[i] if self.N_robust > 1 else self.field

            fig.add_subplot(2,self.N_robust,i+1)
            plt.errorbar(y_true[i][:,0],y_pred[i][:,0],y_std[i][:,0],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(y_true[i][:,0],y_pred[i][:,0],s=50,c='k',zorder=20)
            plt.plot(ideal1,ideal1,"r",lw=3,zorder=40)
            plt.xlabel(r"$\Omega_\mathrm{m, true}$", fontsize=fontsize)
            plt.ylabel(r"$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
            #plt.title(title,fontsize=fontsize)

            fig.add_subplot(2,self.N_robust,i+self.N_robust+1)
            plt.errorbar(y_true[i][:,1],y_pred[i][:,1],y_std[i][:,1],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(y_true[i][:,1],y_pred[i][:,1],s=50,c='k',zorder=20)
            plt.plot(ideal2,ideal2,"r",lw=3,zorder=40)
            plt.xlabel(r"$\sigma_\mathrm{8, true}$", fontsize=fontsize)
            plt.ylabel(r"$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
            #plt.title(title, fontsize=fontsize)

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        if save_plot:
            plt.savefig("img/{}_result.png".format(fname), bbox_inches="tight",
                        dpi=dpi)
        if show_plot is False:
            plt.close()
        if data_return:
            return y_true, y_pred, y_std
        return



    def test_on_cnn(self, sims, fname=None, show_plot=True, show_score=True, data_return=False, save_plot=False):
        self.vib.eval()

        # Data load
        X, y = self.load_data(external=True, external_sims=sims)

        print(X.shape, y.shape)

        ideal1 = np.linspace(0.1,0.5,3)
        ideal2 = np.linspace(0.6,1.0,3)

        if self.robust_axis == 'field':
            fontsize=30
            plt.rcParams['font.size'] = '30'
            plt.rcParams['font.family'] = 'sans-serif'
            #plt.rcParams['font.sans-serif'] = 'sans-serif'
            plt.rcParams['xtick.labelsize'] = '15'
            plt.rcParams['ytick.labelsize'] = '15'
            fig = plt.figure(figsize=(20,20))
            for i in range(self.N_robust):
                pred, std = self.predict_cnn(X[i*15000:(i+1)*15000])
                title = self.field[i] if self.N_robust > 1 else self.field

                fig.add_subplot(2,self.N_robust,i+1)
                plt.errorbar(y[15000*i:15000*(i+1),0],pred[:,0],std[:,0],linestyle="None",ecolor="grey", capsize=3)#, s=1)
                plt.scatter(y[15000*i:15000*(i+1),0],pred[:,0],s=50,c='k',zorder=20)
                plt.plot(ideal1,ideal1,"r",lw=3,zorder=40)
                plt.xlabel(r"$\Omega_\mathrm{m, true}$", fontsize=fontsize)
                plt.ylabel(r"$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
                plt.title(title,fontsize=fontsize)

                fig.add_subplot(2,self.N_robust,i+self.N_robust+1)
                plt.errorbar(y[15000*i:15000*(i+1),1],pred[:,1],std[:,1],linestyle="None",ecolor="grey", capsize=3)#, s=1)
                plt.scatter(y[15000*i:15000*(i+1),1],pred[:,1],s=50,c='k',zorder=20)
                plt.plot(ideal2,ideal2,"r",lw=3,zorder=40)
                plt.xlabel(r"$\sigma_\mathrm{8, true}$", fontsize=fontsize)
                plt.ylabel(r"$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
                plt.title(title, fontsize=fontsize)

        else:
            # Prediction
            pred, std = self.predict_cnn(X)
            title      = sims

            ## Figure settings
            fontsize=30
            plt.rcParams['font.size'] = '30'
            plt.rcParams['font.family'] = 'sans-serif'
            #plt.rcParams['font.sans-serif'] = 'sans-serif'
            plt.rcParams['xtick.labelsize'] = '15'
            plt.rcParams['ytick.labelsize'] = '15'

            fig = plt.figure(figsize=(10,20))

            fig.add_subplot(2,1,1)
            plt.errorbar(y[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey",
                         capsize=3,zorder=0)#, s=1)
            plt.scatter(y[:,0],pred[:,0],s=20,c='k', zorder=10)
            plt.plot(ideal1,ideal1,"r",lw=3,zorder=12)
            plt.xlabel(r"$\Omega_\mathrm{m, true}$", fontsize=fontsize)
            plt.ylabel(r"$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
            plt.title(title, fontsize=fontsize)

            fig.add_subplot(2,1,2)
            plt.errorbar(y[:,1],pred[:,1],std[:,1],linestyle="None",ecolor="grey",
                         capsize=3,zorder=0)#, s=1)
            plt.scatter(y[:,1],pred[:,1],s=20,c='k',zorder=10)
            plt.plot(ideal2,ideal2,"r",lw=3,zorder=12)
            plt.xlabel(r"$\sigma_\mathrm{8, true}$", fontsize=fontsize)
            plt.ylabel(r"$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
            plt.title(title, fontsize=fontsize)

        plt.subplots_adjust(hspace=0.2, wspace=0.4)

        if show_plot is False:
            plt.close()
        if save_plot:
            plt.savefig("img/{}_test_on_{}.png".format(fname, sims), bbox_inches="tight",
                        dpi=100)
        if show_score:
            self.get_score_cnn(test_set=False,X=X,y=y,_print=True)
        if data_return:
            return y, (pred, std)



    def get_score_cnn(self, test_set=True, bias_test=False, X=None, y=None,
                  _print=True):
        # MSE, Relative errors, Bias, AUC for total and each simulation
        self.vib.eval()
        if self.which_machine in self.mach_vib_cls:
            self.cls.eval()

        if test_set:
            X     = torch.tensor(self.input[self.test_indices],dtype=torch.float).to(self.device)
            y     = self.output[self.test_indices]
            if self.which_machine in self.mach_vib_cls:
                y_cls = self.output_cls[self.test_indices]
        elif (X is None) or (y is None):
            print("No data!")
        else:
            X = torch.tensor(X,dtype=torch.float).to(self.device)

        if self.which_machine in self.is_cls:
            y_pred, std, pred_cls = self.predict_cnn(X, cls=True)
        else:
            y_pred, std = self.predict_cnn(X, cls=False)


        param_names  = [r"$\Omega_m$", r"$\sigma_8$"]
        MSE=[]; r2=[]; chi2=[];
        for i in range(2):
            diff = y[:,i]-y_pred[:,i]
            MSE.append(np.power(diff,2).mean())
            rel  = np.abs(diff)/y[:,i]
            rel  = rel.mean()*100
            bias = diff
            bias = bias.mean()
            r2.append(sklearn.metrics.r2_score(y[:,i],y_pred[:,i]))
            chi2.append(np.mean(diff**2/std[:,i]**2))
            if _print:
                print(r"{}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f},chi2={:.3f}, bias={:.3f}"\
                      .format(param_names[i], MSE[-1], rel, r2[-1], chi2[-1], bias))

        if test_set:
            if self.which_machine in self.mach_vib_cls:
                auc = sklearn.metrics.roc_auc_score(y_cls, pred_cls)
                if _print:
                    print("The ROC AUC score for classification is {}.".format(auc))
                    print("")
            else:
                auc = np.inf


        if test_set is False:
            return

        if bias_test:
            for j in range(self.N_robust):
                index      = np.logical_and(self.test_indices<(j+1)*1000, self.test_indices>=j*1000)
                X          = torch.tensor(self.input[self.test_indices[index]],dtype=torch.float).to(self.device)
                true       = self.output[self.test_indices[index]]
                pred, std  = self.predict_cnn(X)
                title      = self.sims[j] if self.N_robust > 1 else self.sims
                for i in range(2):
                    diff = true[:,i]-pred[:,i]
                    MSE_  = np.power(diff,2).mean()
                    rel  = np.abs(diff)/true[:,i]
                    rel  = rel.mean()*100
                    bias = diff #/true[:,i]
                    bias = bias.mean()
                    r2_   = sklearn.metrics.r2_score(true[:,i], pred[:,i])
                    if _print:
                        print("{} of {}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f} bias={:.3f}"\
                              .format(param_names[i], title, MSE_, rel, r2_, bias))
        return MSE[0], MSE[1], chi2[0],chi2[1], np.abs(auc-0.5)**2*1e-2  ## 





    def make_plots(self, fname="test", dpi=100, figsize=(20,20),
                   batch_size=None,
                   show_plot=True, data_return=False, save_plot=True):
        if self.which_machine == 'xgboost'\
           or self.which_machine == 'tabnet':
            pass
        else:
            self.vib.eval()
            if self.which_machine in self.mach_vib_cls:
                self.cls.eval()

        ## Figure settings
        fontsize=30
        plt.rcParams['font.size'] = '30'
        plt.rcParams['font.family'] = 'sans-serif'
        #plt.rcParams['font.sans-serif'] = 'sans-serif'
        plt.rcParams['xtick.labelsize'] = '15'
        plt.rcParams['ytick.labelsize'] = '14'

        fig = plt.figure(figsize=figsize)
        ideal1 = np.linspace(0.1,0.5,3)
        ideal2 = np.linspace(0.6,1.0,3)


        if data_return:
            y_true = []
            y_pred = []
        if self.projection:
            N = 3000
        elif self.average:
            N = 1000
        else:
            N = 15000

        for i in range(self.N_robust):
            index      = np.logical_and(self.test_indices<(i+1)*N,
                                        self.test_indices>=i*N)
            X          = self.input[self.test_indices[index]]
            true       = self.output[self.test_indices[index]]
            if self.which_machine == 'xgboost':
                pred  = self.bst.predict(X)
                if data_return:
                    y_true.append(true)
            elif self.which_machine == 'tabnet':
                pred  = self.rgs.predict(X)
                if data_return:
                    y_true.append(true)
                    y_pred.append(pred)
            else:
                pred, std  =\
                        self.vib(torch.tensor(X,dtype=torch.float).to(self.device))
                pred       = pred.cpu().detach().numpy()
                std        = std.cpu().detach().numpy()
                if data_return:
                    y_true.append(true)
                    y_pred.append([pred, std])
            title      = self.sims[i] if self.num_sim > 1 else self.sims

            fig.add_subplot(2,self.N_robust,i+1)
            if self.which_machine != 'xgboost' \
               and self.which_machine != 'tabnet':
                plt.errorbar(true[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,0],pred[:,0],s=50,c='k',zorder=20)
            plt.plot(ideal1,ideal1,"r",lw=3)
            plt.xlabel("$\Omega_\mathrm{m, true}$", fontsize=fontsize)
            plt.ylabel("$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
            plt.title(title,fontsize=fontsize)

            fig.add_subplot(2,self.N_robust,i+self.N_robust+1)
            if self.which_machine != 'xgboost'\
               and self.which_machine != 'tabnet':
                plt.errorbar(true[:,1],pred[:,1],std[:,1],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,1],pred[:,1],s=50,c='k',zorder=20)
            plt.plot(ideal2,ideal2,"r",lw=3)
            plt.xlabel("$\sigma_\mathrm{8, true}$", fontsize=fontsize)
            plt.ylabel("$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
            plt.title(title, fontsize=fontsize)

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        if save_plot:
            plt.savefig("img/{}_result.png".format(fname), bbox_inches="tight",
                        dpi=dpi)
        if show_plot is False:
            plt.close()
        if data_return:
            return y_true, y_pred

    def test_on(self, sims, fname=None, show_plot=True, show_score=True, data_return=False, save_plot=False):
        if self.which_machine == 'xgboost' \
               or self.which_machine == 'tabnet':
            pass
        else:
            self.vib.eval()

        # Data load
        X, y = self.load_data(external=True, external_sims=sims)
        print(y.shape)

        # Prediction
        if self.which_machine == 'xgboost':
            pred       = self.bst.predict(X)
        elif self.which_machine == 'tabnet':
            pred       = self.rgs.predict(X)
        else:
            pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(self.device))
            pred       = pred.cpu().detach().numpy()
            std        = 3*np.abs(std.cpu().detach().numpy())
        title      = sims


        ## Figure settings
        fontsize=30
        plt.rcParams['font.size'] = '30'
        plt.rcParams['font.family'] = 'sans-serif'
        #plt.rcParams['font.sans-serif'] = 'sans-serif'
        plt.rcParams['xtick.labelsize'] = '15'
        plt.rcParams['ytick.labelsize'] = '15'

        fig = plt.figure(figsize=(10,20))
        ideal1 = np.linspace(0.1,0.5,3)
        ideal2 = np.linspace(0.6,1.0,3)

        fig.add_subplot(2,1,1)
        if self.which_machine != 'xgboost'\
               and self.which_machine != 'tabnet':
            plt.errorbar(y[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey",
                         capsize=3,zorder=0)#, s=1)
        plt.scatter(y[:,0],pred[:,0],s=20,c='k', zorder=10)
        plt.plot(ideal1,ideal1,"r",lw=3,zorder=12)
        plt.xlabel("$\Omega_\mathrm{m, true}$", fontsize=fontsize)
        plt.ylabel("$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
        plt.title(title, fontsize=fontsize)

        fig.add_subplot(2,1,2)
        if self.which_machine != 'xgboost'\
               and self.which_machine != 'tabnet':
            plt.errorbar(y[:,1],pred[:,1],std[:,1],linestyle="None",ecolor="grey",
                         capsize=3,zorder=0)#, s=1)
        plt.scatter(y[:,1],pred[:,1],s=20,c='k',zorder=10)
        plt.plot(ideal2,ideal2,"r",lw=3,zorder=12)
        plt.xlabel("$\sigma_\mathrm{8, true}$", fontsize=fontsize)
        plt.ylabel("$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
        plt.title(title, fontsize=fontsize)

        plt.subplots_adjust(hspace=0.2, wspace=0.4)
        if show_plot is False:
            plt.close()
        if save_plot:
            plt.savefig("img/{}_test_on_{}.png".format(fname, sims), bbox_inches="tight",
                        dpi=100)
        if show_score:
            self.get_score(test_set=False,X=X,y=y,_print=True)
        if data_return:
            return y, (pred, std)





    def get_score(self, test_set=True, bias_test=False, X=None, y=None,
                  _print=True):
        # MSE, Relative errors, Bias, AUC for total and each simulation
        self.vib.eval()
        if self.which_machine in self.mach_vib_cls:
            self.cls.eval()

        if test_set:
            X     = torch.tensor(self.input[self.test_indices],dtype=torch.float).to(self.device)
            y     = self.output[self.test_indices]
            if self.which_machine in self.mach_vib_cls:
                y_cls = self.output_cls[self.test_indices]
        elif (X is None) or (y is None):
            print("No data!")
        else:
            X = torch.tensor(X,dtype=torch.float).to(self.device)

        y_pred, std  = self.vib(X)
        y_pred, std  = y_pred.cpu().detach().numpy(), std.cpu().detach().numpy()
        if self.which_machine in self.mach_vib_cls:
            pred_cls     = self.cls(self.vib.get_latent_variable())

        param_names  = [r"$\Omega_m$", r"$\sigma_8$"]
        MSE=[]; r2=[]; chi2=[];
        for i in range(2):
            diff = y[:,i]-y_pred[:,i]
            MSE.append(np.power(diff,2).mean())
            rel  = np.abs(diff)/y[:,i]
            rel  = rel.mean()*100
            bias = diff #/y[:,i]
            bias = bias.mean()
            r2.append(sklearn.metrics.r2_score(y[:,i],y_pred[:,i]))
            chi2.append(np.mean(diff**2/std[:,i]**2))
            if _print:
                print(r"{}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f},chi2={:.3f}, bias={:.3f}"\
                      .format(param_names[i], MSE[-1], rel, r2[-1], chi2[-1], bias))

        if test_set:
            if self.which_machine in self.mach_vib_cls:
                auc = sklearn.metrics.roc_auc_score(y_cls, pred_cls.detach().cpu().numpy())
                if _print:
                    print("The ROC AUC score for classification is {}.".format(auc))
                    print("")
            else:
                auc = np.inf


        if test_set is False:
            return

        if bias_test:
            for j in range(self.N_robust):
                index      = np.logical_and(self.test_indices<(j+1)*1000, self.test_indices>=j*1000)
                X          = torch.tensor(self.input[self.test_indices[index]],dtype=torch.float).to(self.device)
                true       = self.output[self.test_indices[index]]
                pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
                pred       = pred.cpu().detach().numpy()
                std        = std.cpu().detach().numpy()
                title      = self.sims[j] if self.N_robust > 1 else self.sims
                for i in range(2):
                    diff = true[:,i]-pred[:,i]
                    MSE_  = np.power(diff,2).mean()
                    rel  = np.abs(diff)/true[:,i]
                    rel  = rel.mean()*100
                    bias = diff #/true[:,i]
                    bias = bias.mean()
                    r2_   = sklearn.metrics.r2_score(true[:,i], pred[:,i])
                    if _print:
                        print("{} of {}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f} bias={:.3f}"\
                              .format(param_names[i], title, MSE_, rel, r2_, bias))
        return MSE[0], MSE[1], chi2[0],chi2[1], np.abs(auc-0.5)**2*1e-2  ## 

    def __process_imag_to_phase(self,_input):
        _radius = np.absolute(_input)
        _phase  = np.angle(_input)
        _input_new = np.empty((_input.shape[0],_input.shape[1]*2),dtype=float)
        _input_new[:,:_radius.shape[1]] = _radius
        _input_new[:,_radius.shape[1]:] = _phase
        return _input_new





