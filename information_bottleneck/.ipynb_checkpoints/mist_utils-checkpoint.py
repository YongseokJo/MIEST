import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
import vib_utils
import warnings
warnings.filterwarnings('ignore')



class MIST():
    def __init__(self, sim="TNG", field="Mtot", batch_size=100, extended_L=False, device='cuda'):
        self.sims       = sim
        self.field      = field
        self.batch_size = batch_size
        self.device     = device
        self.num_sim    = len(self.sims) if isinstance(self.sims, list) else 1
        
        self.input, self.output = self.load_data(extended_L=False)
        self.make_train_set()
        
        
    def train(self,which_machine="vib+cls",fname="test",beta=1e-3, learning_rate=1e-3,
              decay_rate=0.97, z_dim=200, epochs=3000):
        if which_machine == "vib+cls":
            machine      = VIB_NDR
            train_module = train_vib_ndr
        elif which_machine == "vib":
            mahicne      = VIB
            train_module = train_vib
            
        self.vib = machine(self.input.shape[1], self.output.shape[1],
                       z_dim, num_models=self.num_sim)
        self.cls = classifier(z_dim, num_models=self.num_sim)

        _, _ = train_module(self.vib, self.cls, self.train_loader, fname,
                             self.device, epochs,self.batch_size,self.test_dataset,beta=beta)
        #print_score(test_dataset,vib,field)
            
    def make_train_set(self,imaginary=False):
        validation_split = .2
        shuffle_dataset  = True
        random_seed      = 42

        # Labelling for simultions
        y_params   = torch.tensor(self.output,dtype=torch.float)
        y          = torch.zeros((y_params.shape[0],y_params.shape[1]+self.num_sim))
        y[:,:y_params.shape[1]]    = y_params
        for i in range(self.num_sim):
            y[i*1000:(i+1)*1000 ,y_params.shape[1]+i] = 1.
        self.output_cls = y[:,-self.num_sim:]


        X = torch.tensor(np.absolute(self.input),dtype=torch.float)
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
        
        
    def save_models(self,fname="test"):
        torch.save(self.vib.state_dict(),"./model/{}_vib.pt".format(fname))
        torch.save(self.cls.state_dict(),"./model/{}_cls.pt".format(fname))
    
    
    def load_models(self,fname="test", which_machine="vib+cls", z_dim=200):
        if which_machine == "vib+cls":
            machine      = VIB_NDR
        elif which_machine == "vib":
            mahicne      = VIB
        self.vib = machine(self.input.shape[1], self.output.shape[1],
                           z_dim, num_models=self.num_sim)
        self.cls = classifier(z_dim, num_models=self.num_sim)
        self.vib.load_state_dict(torch.load("./model/{}_vib.pt".format))
        self.cls.load_state_dict(torch.load("./model/{}_cls.pt".format))
        self.vib.eval()
        self.cls.eval()    
        
            
    def load_data(self, extended_L=False):
        fparams = {"TNG":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_IllustrisTNG.txt',
                   "SIMBA":'/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/params_SIMBA.txt',
                   "ASTRID":"/mnt/home/fvillaescusa/CAMELS/Results/images_Astrid/params_LH_Astrid.txt",
                   "RAMSES":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Ramses/params_Ramses.txt",
                   "GADGET":"/mnt/ceph/users/fvillaescusa/Nbody_systematics/data/maps/maps_Gadget/params_Gadget.txt",
                  }
        sim_names = {"TNG":"IllustrisTNG",
                    "SIMBA":"SIMBA",
                    "ASTRID":"Astrid",
                    "GADGET": "Gadget",
                    "RAMSES": "Ramses",}
        
        if isinstance(self.sims, list):
            coefs  = []
            params = []
            for sim in self.sims:
                coef = np.load("/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_n{}_{}_for_vib_total.npy"\
                               .format(sim_names[sim], self.field))
                coef_avg = np.zeros((1000, coef.shape[1]))
                for i in range(1000):
                    coef_avg[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                coefs.append(coef_avg)     
                
                param = np.loadtxt(fparams[sim])
                params.append(param[:,:2])  ## only Om and Sig8
                
            _input  = np.vstack(coefs)
            _output = np.vstack(params)
                

        else:
            coef = np.load("/mnt/home/yjo10/ceph/CAMELS/MIEST/data/wph_n{}_{}_for_vib_total.npy"\
                               .format(sim_names[self.sims], self.field))
            _input = np.zeros((1000, coef.shape[1]))
            for i in range(1000):
                _input[i,:] = coef[i*15:i*15+15,:].mean(axis=0)
                
            _output = np.loadtxt(fparams[self.sims])
        return _input, _output
            
    def make_plots(self, fname="test"):
        self.vib.eval()
        self.cls.eval()
        
        ## Figure settings
        fontsize=50
        plt.rcParams['font.size'] = '50'
        plt.rcParams['font.family'] = 'sans-serif'
        #plt.rcParams['font.sans-serif'] = 'sans-serif'
        plt.rcParams['xtick.labelsize'] = '30'
        plt.rcParams['ytick.labelsize'] = '30'

        fig = plt.figure(figsize=(30,20))
        ideal1 = np.linspace(0.1,0.5,3)
        ideal2 = np.linspace(0.6,1.0,3)
        
        
        for i in range(self.num_sim):
            index      = np.logical_and(self.val_indices<(i+1)*1000, self.val_indices>=i*1000)
            X          = self.input[self.val_indices[index]]
            true       = self.output[self.val_indices[index]]
            pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
            pred       = pred.cpu().detach().numpy()
            std        = std.cpu().detach().numpy()
            title      = self.sims[i] if self.num_sim > 1 else self.sims
            
            fig.add_subplot(2,self.num_sim,i+1)
            plt.errorbar(true[:,0],pred[:,0],std[:,0],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,0],pred[:,0],s=50,c='k')
            plt.plot(ideal1,ideal1,"r",lw=3)
            plt.xlabel("$\Omega_\mathrm{m, true}$", fontsize=fontsize)
            plt.ylabel("$\Omega_\mathrm{m, pred}$", fontsize=fontsize)
            plt.title(title)
            
            fig.add_subplot(2,self.num_sim,i+self.num_sim+1)
            plt.errorbar(true[:,1],pred[:,1],std[:,1],linestyle="None",ecolor="grey", capsize=3)#, s=1)
            plt.scatter(true[:,1],pred[:,1],s=50,c='k')
            plt.plot(ideal2,ideal2,"r",lw=3)
            plt.xlabel("$\sigma_\mathrm{8, true}$", fontsize=fontsize)
            plt.ylabel("$\sigma_\mathrm{8, pred}$", fontsize=fontsize)
            plt.title(title, fontsize=fontsize)

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig("img/{}_result.png".format(fname), bbox_inches="tight", dpi=300)

        
    def test_on_diff_sim(self, sim):
        pass
    
    def print_score(self,):
        # MSE, Relative errors, Bias, AUC for total and each simulation
        self.vib.eval()
        self.cls.eval()
        
        X            = torch.tensor(self.input[self.val_indices],dtype=torch.float).to(self.device)
        y_param      = self.output[self.val_indices]
        y_cls        = self.output_cls[self.val_indices]
        y_pred, std  = self.vib(X)
        pred_cls     = self.cls(self.vib.get_latent_variable())
        
        param_names  = [r"$\Omega_m$", r"$\sigma_8$"]
        for i in range(2):
            diff = y_param[:,i]-y_pred[:,i].cpu().detach().numpy()
            MSE  = np.power(diff,2).mean()
            rel  = np.abs(diff)/y_param[:,i]
            rel  = rel.mean()*100
            bias = diff/y_param[:,i]
            bias = bias.mean()
            r2   = sklearn.metrics.r2_score(y_param[:,i], y_pred[:,i].cpu().detach().numpy())
            print(r"{}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f} bias={:.3f}".format(param_names[i], MSE, rel, r2, bias))
            
        auc  = sklearn.metrics.roc_auc_score(
                y_cls, pred_cls.detach().cpu().numpy())
        print("The ROC AUC score for classification is {}.".format(auc))
        print("")
        
        for j in range(self.num_sim):
            index      = np.logical_and(self.val_indices<(j+1)*1000, self.val_indices>=j*1000)
            X          = torch.tensor(self.input[self.val_indices[index]],dtype=torch.float).to(self.device)
            true       = self.output[self.val_indices[index]]
            pred, std  = self.vib(torch.tensor(X,dtype=torch.float).to(device))
            pred       = pred.cpu().detach().numpy()
            std        = std.cpu().detach().numpy()
            title      = self.sims[i] if self.num_sim > 1 else self.sims
            for i in range(2):
                diff = true[:,i]-pred[:,i]
                MSE  = np.power(diff,2).mean()
                rel  = np.abs(diff)/true[:,i]
                rel  = rel.mean()*100
                bias = diff/true[:,i]
                bias = bias.mean()
                r2   = sklearn.metrics.r2_score(true[:,i], pred[:,i])
                print("{} of {}: MSE={:.3f}, % error={:.3f}%, R2 score={:.3f} bias={:.3f}"\
                      .format(param_names[j], title, MSE, rel, r2, bias))
            
        
        
    
            