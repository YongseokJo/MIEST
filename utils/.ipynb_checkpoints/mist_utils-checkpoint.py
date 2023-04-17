import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
import sklearn
import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from vib_utils import *
import warnings
warnings.filterwarnings('ignore')



class MIST():
    self.mach_vib__cls = ['vib+cls', 'vib+cls_a']
    self.mach_vib      = ['vib', 'vib_cnn']
    self.mach_no_vib   = ['fcl', 'cnn']
    
    def __init__(self, sim="TNG", field="Mtot", batch_size=100, 
                 extended_L=None, extended_dn=None, 
                 normalization=False, monopole=True,
                 average=True, device='cuda'):
        self.sims          = sim
        self.field         = field
        self.batch_size    = batch_size
        self.normalization = normalization
        self.monopole      = monopole
        self.average       = average
        self.device        = device
        self.num_sim       = len(self.sims) if isinstance(self.sims, list) else 1

        self.input, self.output = self.load_data(extended_L=extended_L,
                                                 extended_dn=extended_dn)
        self.process_imaginary  = np.absolute
        self.input              = self.process_imaginary(self.input)
        self.make_train_set()

    def train(self,which_machine="vib+cls",fname="test",beta=1, gamma=1e-2, learning_rate=1e-3,
              decay_rate=0.97, z_dim=200, epochs=3000, hidden=1, dr=0.5,channels=1,
              verbose=True, save_plot=True, save_model=True):
        if which_machine == 'fcl':
            self.vib = FCL(self.input.shape[1], self.output.shape[1])
            self.cls = None
        elif which_machine == 'cnn':
            self.vib = cnn(hidden=hidden, dr=dr,channels=channels)
            self.cls = None
        elif which_machine == 'vib':
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim)
            self.cls = None
        elif which_machine == 'vib_cnn':
            self.vib = VIB_CNN(hidden=hidden, dr=dr,channels=channels)
            self.cls = None
        elif which_machine == "vib+cls":
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim)
            self.cls = classifier(z_dim, num_models=self.num_sim)
        elif which_machine == "vib+cls_a":
            self.vib = VIB_a(input_shape=self.input.shape[1],
                             output_shape=self.output.shape[1],
                             z_dim=z_dim, h=hidden, dr=dr)
            self.cls = classifier(z_dim, num_models=self.num_sim)

        trainer = Trainer(self.vib, self.cls, self.train_loader, which_machine,
                          self.test_dataset,fname,self.device)
        res     = trainer.run(learning_rate, epochs, decay_rate, beta, gamma,
                              verbose, save_plot=save_plot, save_model=save_model)
        self.vib, self.cls, success = res
        return success

    def make_train_set(self,imaginary=False):
        validation_split = .2
        shuffle_dataset  = True
        random_seed      = 42

        # Labelling for simultions
        y_params   = torch.tensor(self.output,dtype=torch.float)
        y          = torch.zeros((y_params.shape[0],y_params.shape[1]+self.num_sim))
        y[:,:y_params.shape[1]]    = y_params
        if self.average:
            N = 1000
        else:
            N = 15000
        for i in range(self.num_sim):
            y[i*N:(i+1)*N ,y_params.shape[1]+i] = 1.
        self.output_cls = y[:,-self.num_sim:]


        X = torch.tensor(self.input,dtype=torch.float)
        dataset      = data_utils.TensorDataset(X, y)
        dataset_size = len(dataset)
        indices      = list(range(dataset_size))
        split        = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, self.val_indices = indices[split:], np.array(indices[:split])

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(self.val_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                        sampler=train_sampler)
        self.test_dataset = data_utils.TensorDataset(X[self.val_indices], y[self.val_indices])
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                                       sampler=valid_sampler)


    def save_models(self,fname="test",fpath=None):
        if fpath is None:
            fpath = "./model"
        torch.save(self.vib.state_dict(),"{}/{}_vib.pt".format(fpath, fname))
        if self.which_machine != 'vib':
            torch.save(self.cls.state_dict(),"{}/{}_cls.pt".format(fpath, fname))


    def load_models(self,fname="test", which_machine="vib+cls", 
                    z_dim=200, hidden=1, dr=0.5, channels=1):
        self.which_machine = which_machine
        ## I have to add multiple inputs
        if which_machine == 'fcl':
            self.vib = FCL(self.input.shape[1], self.output.shape[1])
        elif which_machine == 'cnn':
            self.vib = cnn(hidden=hidden, dr=dr,channels=channels)
        elif which_machine == 'vib_cnn':
            self.vib = VIB_CNN(hidden=hidden, dr=dr,channels=channels)
        elif which_machine == 'vib':
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim)
        elif which_machine == 'vib+cls':
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim)
        elif which_machine == 'vib+cls_a':
            self.vib = VIB_a(self.input.shape[1], self.output.shape[1],
                           z_dim=z_dim, h=hidden, dr=dr)

        self.vib.load_state_dict(torch.load("./model/{}_vib.pt".format(fname)))
        self.vib.to(self.device).eval()
        if which_machine in self.mach_vib_cls:
            self.cls = classifier(z_dim, num_models=self.num_sim)
            self.cls.load_state_dict(torch.load("./model/{}_cls.pt".format(fname)))
            self.cls.to(self.device).eval()


    def load_optuna_models(self, storage, study_name, metric=None, suffix='',
                           which_machine="vib+cls_a", num_trial=None,
                           print_loss=True):
        self.which_machine = which_machine
        import optuna_utils as op
        fpath = './model/optuna/'
        num_trial, params = op.best_params(study_name, storage,
                                           verbose=False,
                                           metric=metric,
                                           num_trial=num_trial)
        if self.which_machine == 'vib':
            fname = '{}_{}_vib'.format(self.field, num_trial)
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'])
            self.cls = None
        elif which_machine == 'vib+cls':
            fname = '{}_{}'.format(self.field, num_trial)
            self.vib = VIB(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'])
            self.cls = classifier(z_dim=params['z_dim'], num_models=self.num_sim)
        elif which_machine == 'vib+cls_a' or which_machine == 'vib+cls_a':
            fname = '{}_{}_a'.format(self.field, num_trial)
            self.vib = VIB_a(self.input.shape[1], self.output.shape[1],
                           z_dim=params['z_dim'],h=params['hidden'],
                           dr=params['dropout'])
            self.cls = classifier(z_dim=params['z_dim'], num_models=self.num_sim)

        self.vib.load_state_dict(torch.load("{}/{}_{}vib.pt".format(fpath,fname,suffix)))
        self.vib.to(self.device).eval()
        if self.which_machine in self.mach_vib_cls:
            self.cls.load_state_dict(torch.load("{}/{}_{}cls.pt".format(fpath,fname,suffix)))
            self.cls.to(self.device).eval()
        if print_loss:
            print("beta={}, gamma={}".format(params['beta'], params['gamma']))


    def load_data(self, data_type="wph", extended_L=None, extended_dn=None, external=False, external_sims=None,):
        sim_names = {"TNG":"IllustrisTNG",
                     "SIMBA":"SIMBA",
                     "ASTRID":"Astrid",
                     "GADGET": "Gadget",
                     "RAMSES": "Ramses",}

        if data_type == 'image'
            field="T"
    fmaps = \
    "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_SIMBA_LH_z=0.00.npy".format(field)
    # read the data
    simba_maps = np.log10(np.load(fmaps))
    fmaps = \
    "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_IllustrisTNG_LH_z=0.00.npy".format(field)
    # read the data
    tng_maps = np.log10(np.load(fmaps))
    fmaps = \
    "/mnt/ceph/users/camels/Results/images_Astrid/Images_{}_Astrid_LH_z=0.00.npy".format(field)
    ast_maps = np.log10(np.load(fmaps))

    fparams = {"TNG":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_IllustrisTNG.txt',
               "SIMBA":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_SIMBA.txt',
               "ASTRID":"/mnt/home/fvillaescusa/CAMELS/Results/images_Astrid/params_LH_Astrid.txt"}
    
    maps_tng   = tng_maps.reshape(15000,1,256,256)
    params_tng = np.loadtxt(fparams['TNG'])[:,:2]
    params_tng_extended = np.ones((15000,2))
    for i in range(1000):
        params_tng_extended[i*15:(i+1)*15,:] = params_tng[i,:]

    maps_simba   = simba_maps.reshape(15000,1,256,256)
    params_simba = np.loadtxt(fparams['SIMBA'])[:,:2]
    params_simba_extended = np.ones((15000,2))
    for i in range(1000):
        params_simba_extended[i*15:(i+1)*15,:] = params_simba[i,:]
        
        
    maps = np.r_[maps_tng,maps_simba]
    params = np.r_[params_tng_extended, params_simba_extended]
    #maps = maps_tng
    #params = params_tng_extended
    #maps = maps_simba
    #params = params_simba_extended
    #maps = maps_tng[::15]
    #params = params_tng
    #nmaps = (maps-maps.mean())/maps.std()
    #nparams = (params-np.array([0.1,0.6]))/(np.array([0.5,1.0])-np.array([0.1,0.6]))


        elif data_type == 'wph'
            suffix = ""
            if extended_L == 10:
                suffix = "_l_10"
                if extended_dn == 2:
                    suffix = "_l_10_dn_2"
            fparams = {"TNG":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_IllustrisTNG.txt',
                       "SIMBA":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_SIMBA.txt',
                       "ASTRID":"/mnt/home/fvillaescusa/CAMELS/Results/images_Astrid/params_LH_Astrid.txt",
                       "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/params_Ramses.txt",
                       "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/params_Gadget.txt",
                      }


            if external:
                if external_sims is None:
                    print("You need to pass simulation type!")
                else:
                    sims = external_sims
            else:
                sims  = self.sims

            prefix = '' if self.monopole else 'n'
            if isinstance(sims, list):
                coefs  = []
                params = []
                for sim in sims:
                    coef = np.load(
                        "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}{}_{}_for_vib_total{}.npy"\
                        .format(prefix,sim_names[sim], self.field, suffix))
                    if self.average:
                        coef_avg = np.zeros((1000, coef.shape[1]))
                        for i in range(1000):
                            coef_avg[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                        coefs.append(coef_avg)
                        param = np.loadtxt(fparams[sim])
                        params.append(param[:,:2])  ## only Om and Sig8
                    else:
                        coefs.append(coef)
                        param     = np.loadtxt(fparams[sim])[:,:2]
                        param_ext = np.ones((15000,2))
                        for i in range(1000):
                            param_ext[i*15:(i+1)*15,:] = \
                                    param[i,:]
                        params.append(param_ext)  ## only Om and Sig8

                _input  = np.vstack(coefs)
                _output = np.vstack(params)
            else:
                coef = np.load(
                    "/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_{}{}_{}_for_vib_total{}.npy"\
                    .format(prefix,sim_names[sims], self.field,suffix))
                if self.average:
                    _input = np.zeros((1000, coef.shape[1]))
                    for i in range(1000):
                        _input[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                        _output = np.loadtxt(fparams[sims])[:,:2]
                else:
                    _input  = coef
                    param   = np.loadtxt(fparams[sims])[:,:2]
                    _output = np.ones((15000,2))
                    for i in range(1000):
                        _output[i*15:(i+1)*15,:] = param[i,:]

        if self.normalization:
            if external:
                _input = (_input -self.mean_norm)/self.std_norm
            else:
                self.mean_norm = _input.mean()
                self.std_norm  = _input.std()
                _input = (_input -self.mean_norm)/self.std_norm

        return _input, _output


    def make_plots(self, fname="test", dpi=100, figsize=(20,20),show_plot=True, data_return=False, save_plot=True):
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
        if self.average == True:
            N = 1000
        else:
            N = 15000

        for i in range(self.num_sim):
            index      = np.logical_and(self.val_indices<(i+1)*N,
                                        self.val_indices>=i*N)
            X          = self.input[self.val_indices[index]]
            true       = self.output[self.val_indices[index]]
            pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
            pred       = pred.cpu().detach().numpy()
            std        = std.cpu().detach().numpy()
            title      = self.sims[i] if self.num_sim > 1 else self.sims
            if data_return:
                y_true.append(true)
                y_pred.append([pred, std])

            fig.add_subplot(2,self.num_sim,i+1)
            plt.errorbar(true[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,0],pred[:,0],s=50,c='k')
            plt.plot(ideal1,ideal1,"r",lw=3)
            plt.xlabel("$\Omega_\mathrm{m, true}$", fontsize=fontsize)
            plt.ylabel("$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
            plt.title(title,fontsize=fontsize)

            fig.add_subplot(2,self.num_sim,i+self.num_sim+1)
            plt.errorbar(true[:,1],pred[:,1],std[:,1],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,1],pred[:,1],s=50,c='k')
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

    def test_on(self, sims, fname, show_plot=True, show_score=True, data_return=False, save_plot=False):
        self.vib.eval()

        # Data load
        _input, _output = self.load_data(external=True, external_sims=sims)

        # Prediction
        X          = self.process_imaginary(_input)
        y          = _output
        pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
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
        plt.errorbar(y[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey",
                     capsize=3,zorder=0)#, s=1)
        plt.scatter(y[:,0],pred[:,0],s=20,c='k', zorder=10)
        plt.plot(ideal1,ideal1,"r",lw=3,zorder=12)
        plt.xlabel("$\Omega_\mathrm{m, true}$", fontsize=fontsize)
        plt.ylabel("$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
        plt.title(title, fontsize=fontsize)

        fig.add_subplot(2,1,2)
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
            X     = torch.tensor(self.input[self.val_indices],dtype=torch.float).to(self.device)
            y     = self.output[self.val_indices]
            if self.which_machine in self.mach_vib_cls:
                y_cls = self.output_cls[self.val_indices]
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
            bias = diff/y[:,i]
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
            for j in range(self.num_sim):
                index      = np.logical_and(self.val_indices<(j+1)*1000, self.val_indices>=j*1000)
                X          = torch.tensor(self.input[self.val_indices[index]],dtype=torch.float).to(self.device)
                true       = self.output[self.val_indices[index]]
                pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
                pred       = pred.cpu().detach().numpy()
                std        = std.cpu().detach().numpy()
                title      = self.sims[j] if self.num_sim > 1 else self.sims
                for i in range(2):
                    diff = true[:,i]-pred[:,i]
                    MSE_  = np.power(diff,2).mean()
                    rel  = np.abs(diff)/true[:,i]
                    rel  = rel.mean()*100
                    bias = diff/true[:,i]
                    bias = bias.mean()
                    r2_   = sklearn.metrics.r2_score(true[:,i], pred[:,i])
                    if _print:
                        print("{} of {}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f} bias={:.3f}"\
                              .format(param_names[i], title, MSE_, rel, r2_, bias))
        return MSE[0], MSE[1], chi2[0],chi2[1], np.abs(auc-0.5)**2*1e-2  ## 

