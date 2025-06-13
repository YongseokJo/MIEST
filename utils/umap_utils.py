import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys,os

sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from vib_utils import *
from mist_utils import *

sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/myutils/"))
from plt_utils import generateAxesForMultiplePlots, remove_inner_axes

import umap
import umap.plot
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score



# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seeds for reproducibility
seed = 73
np.random.seed(seed)



def do_classification(mist):
    X,y              = mist.train_loader.dataset.tensors
    X_train, y_train = X[mist.train_indices].cpu().detach().numpy(), y[mist.train_indices].cpu().detach().numpy()
    y_train_param    = y_train[:,:2]
    y_train          = y_train[:,2:]
    Z_train          = mist.get_latent_variable(X=X_train)
    #Z_train          = mist.get_mu(X=X_train)

    clf = RandomForestClassifier(n_estimators=50,max_depth=20, random_state=0,
                                oob_score=True)
    #clf = RandomForestClassifier(n_estimators=2000,max_depth=None,
    #                             random_state=0)
    #print(Z_train, y_train)
    clf.fit(Z_train, y_train)
    X_test, y_test = mist.test_dataset.tensors
    #Z_test         = mist.get_mu()
    Z_test         = mist.get_latent_variable()
    y_test_param   = y_test[:,:2].cpu().detach().numpy()
    y_test         = y_test[:,2:].cpu().detach().numpy()
    y_pred         = clf.predict(Z_test)
    auc            = sklearn.metrics.roc_auc_score(y_test, y_pred)
    fi             = clf.feature_importances_


    # Perform 5-fold cross-validation
    #scores = cross_val_score(clf, Z_train, y_train, cv=5)

    #oob_score      = clf.oob_score_

    # Get the predicted probabilities
    # Compute log-loss (as an approximation to model evidence)
    #log_likelihood = -log_loss(y_train, y_prob)

    # Get the predicted probabilities
    #print(Z_test.shape, y_test.shape)
    y_prob         = clf.predict_proba(Z_test)
    #print(y_prob[0][:,0], y_test[:,0])
    ##print(y_prob[0]*y_test)

    prob1 = (y_prob[0][:,0]*y_test[:,0]).sum()/y_test[:,0].sum() #.mean()*2
    prob2 = (y_prob[0][:,1]*y_test[:,0]).sum()/y_test[:,0].sum() #.mean()*2
    prob3 = (y_prob[0][:,0]*y_test[:,1]).sum()/y_test[:,1].sum() #.mean()*2
    prob4 = (y_prob[0][:,1]*y_test[:,1]).sum()/y_test[:,1].sum() #.mean()*2

    #print((y_prob[0][:,0]*y_test[:,0]).mean()*2)
    #print((y_prob[0][:,1]*y_test[:,0]).mean()*2)
    #print((y_prob[0][:,0]*y_test[:,1]).mean()*2)
    #print((y_prob[0][:,1]*y_test[:,1]).mean()*2)
    #print(fi)
    #raise
    # Compute log-loss (as an approximation to model evidence)
    #log_likelihood_pred = -log_loss(y_train, y_prob)

    return clf, auc, [prob1, prob2, prob3, prob4],fi, (Z_test, y_test, y_pred, y_test_param),\
            (Z_train, y_train, y_train_param)



class UMAP():
    colors  = ['b', 'r', 'g', 'm']
    markers = ['.', 'X', '*', '^']
    #markers = ['_', '|', '*', '^']
    zorders = [30, 30, 20, 20]
    def __init__(self, sim, field, data_type='image',
                 which_machine='cnn_enc_dec',
                 study_name=None, monopole=True, num_trial=None,
                 fpath=None, ext_sim=None, robust_axis='sim', z_dim=None):
        self.sim     = sim
        self.ext_sim = ext_sim
        storage    =\
                f"sqlite:////mnt/home/yjo10/ceph/CAMELS/MIEST/information_bottleneck/database/{study_name}.db"
        if data_type == 'image':
            self.mist = MIST(sim=sim, field=field, 
                             batch_size=32, 
                             data_type='image',
                             normalization=True,
                             monopole = monopole,
                             device=device, 
                             average=False,
                             projection=False,
                             robust_axis='sim'
                            )
        elif data_type == 'wph':
            self.mist = MIST(sim=sim, field=field, 
                             batch_size=1000, 
                             data_type='wph',
                             normalization=True,
                             monopole = monopole,
                             projection =False,
                             average = True,
                             L=4, dn=2, proc_imag=np.absolute,
                             device=device)
        if z_dim is None:
            self.mist.load_optuna_models(storage=storage,study_name=study_name,
                                         which_machine=which_machine,num_trial=num_trial,
                                         fpath=fpath)
        else:
            self.mist.load_optuna_models(storage=storage,study_name=study_name,
                                         which_machine=which_machine,num_trial=num_trial,
                                         fpath=fpath,z_dim=z_dim)
        self.clf, self.auc, self.prob,  self.fi, self.testset, self.trainset =\
                do_classification(self.mist)
        #print(f"The AUC score is {self.auc}.")
        #print(f"The cross score is {self.cross}.")
        #print(f"The oob score is {self.oob}.")
        #print(f"The log_prob is {self.log_prob}.")
        #print(f"The log_prob_pred is {self.log_prob_pred}.")
        self.Z_test, self.y_test, self.y_pred, self.y_test_param = self.testset
        self.Z_train, self.y_train, self.y_train_param           = self.trainset

        self.clabel = np.empty((self.y_test.shape[0]),dtype=str)
        if robust_axis == 'sim':
            self.robust_len = len(sim)
        elif robust_axis == 'field':
            self.robust_len = len(field)
        else:
            self.robust_len = 1

        print(self.y_test.shape)
        for i in range(self.robust_len):
            self.clabel[self.y_test[:,i]==1] = self.colors[i]

        self.ylabel                       = np.empty((self.y_train.shape[0]),dtype=str)
        for i in range(self.robust_len):
            self.ylabel[self.y_train[:,i]==1] = self.colors[i]

        if ext_sim is not None:
            self.__include_ExtSim()


    def __include_ExtSim(self):
        self.X_ast, self.y_ast = self.mist.load_data(external=True,
                                                     external_sims=self.ext_sim)
        ast_indices = np.random.randint(low=0, high=self.X_ast.shape[0],
                                        size=self.Z_test.shape[0]//2)

        #self.Z_ast = self.mist.get_latent_variable(self.X_ast)
        self.Z_ast = self.mist.get_mu(self.X_ast)
        self.X_ast_test, self.y_ast_test = self.X_ast[ast_indices,:], self.y_ast[ast_indices,:]
        self.Z_ast_test = self.Z_ast[ast_indices,:]

        self.Z_test_all = np.r_[self.Z_test,self.Z_ast_test]
        self.y_test_all = np.zeros((self.Z_test_all.shape[0], len(self.sim)+1))
        self.y_test_all[:self.Z_test.shape[0],:2] = self.y_test
        self.y_test_all[self.Z_test.shape[0]:,2]  = 1 

        self.y_test_param_all = np.zeros((self.Z_test_all.shape[0],len(self.sim)))
        self.y_test_param_all[:self.Z_test.shape[0],:] = self.y_test_param
        self.y_test_param_all[self.Z_test.shape[0]:,:] = self.y_ast_test

        label                                     = self.y_test_all
        self.clabel_all                           = np.empty((label.shape[0]),dtype=str)
        self.clabel_all[label[:,0]==1]            = 'b'
        self.clabel_all[label[:,1]==1]            = 'r'
        self.clabel_all[label[:,2]==1]            = 'g'

        self.cmarker_all                           = np.empty((label.shape[0]),dtype=str)
        self.cmarker_all[label[:,0]==1]            = 'o'
        self.cmarker_all[label[:,1]==1]            = 's'
        self.cmarker_all[label[:,2]==1]            = '*'

        self.Z_train_all = np.r_[self.Z_train,self.Z_ast]

        self.y_train_all = np.zeros((self.Z_train_all.shape[0], len(self.sim)+1))
        self.y_train_all[:self.Z_train.shape[0],:2] = self.y_train
        self.y_train_all[self.Z_train.shape[0]:,2]  = 1 

        self.y_train_param_all = np.zeros((self.Z_train_all.shape[0],len(self.sim)))
        self.y_train_param_all[:self.Z_train.shape[0],:] = self.y_train_param
        self.y_train_param_all[self.Z_train.shape[0]:,:] = self.y_ast

        self.ylabel_all                             = np.empty((self.y_train_all.shape[0]),dtype=str)
        self.ylabel_all[self.y_train_all[:,0]==1]   = 'b'
        self.ylabel_all[self.y_train_all[:,1]==1]   = 'r'
        self.ylabel_all[self.y_train_all[:,2]==1]   = 'g'

        self.ymarker_all                            = np.empty((self.y_train_all.shape[0]),dtype=str)
        self.ymarker_all[self.y_train_all[:,0]==1]  = 'o'
        self.ymarker_all[self.y_train_all[:,1]==1]  = 's'
        self.ymarker_all[self.y_train_all[:,2]==1]  = '*'


    def plot_umap(self,n_neighbors=40,min_dist=0.1, n_components=2,
                  dens_map=False, paint_om=False, paint_sg=False, dens_lambda=2.0,
                  random_state=3, ext_sim=False, connectivity=False,
                  fname=None, return_data=False):

        s = 200
        if ext_sim:
            if dens_map:
                embedding = umap.UMAP(
                    densmap=True, dens_lambda=dens_lambda,
                    random_state=random_state)
            else:
                embedding = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=n_components,
                                      random_state=random_state)
            mapper    = embedding.fit(self.Z_test_all)
            umap_pred = embedding.transform(self.Z_test_all)

            # Paint Omega_m
            if paint_om:
                fig = plt.figure(figsize=(20,20))
                for i in range(3):
                    index = self.clabel_all == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_test_param_all[index,0],
                                marker=self.markers[i],
                                s=s, cmap='Spectral', zorder=self.zorders[i]);
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_om_test_ext.png", dpi=200,bbox_inches='tight')
                plt.show()

            # Paint sigma_8
            if paint_sg:
                fig = plt.figure(figsize=(20,20))
                for i in range(3):
                    index = self.clabel_all == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_test_param_all[index,1],
                                marker=self.markers[i],
                                s=s, cmap='Spectral', zorder=self.zorders[i]);
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_sg_test_ext.png", dpi=200,bbox_inches='tight')
                plt.show()

            fig = plt.figure(figsize=(20,20))
            plt.scatter(umap_pred[:, 0], umap_pred[:, 1],
                        c=self.clabel_all, s=s, cmap='Spectral');
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if fname is not None:
                plt.savefig(f"img/umap_{fname}_test_ext.png", dpi=200,bbox_inches='tight')
            plt.show()

            # Connectivity
            if connectivity:
                fig = plt.figure(figsize=(20,20))
                umap.plot.connectivity(mapper, labels=self.clabel_all, show_points=True)
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.show()

            # Dense map
            if dens_map:
                embedding = umap.UMAP(
                    densmap=True, dens_lambda=dens_lambda,
                    random_state=random_state)
            else:
                embedding = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=n_components,
                                      random_state=random_state)
            mapper    = embedding.fit(self.Z_train_all)
            umap_pred = embedding.transform(self.Z_train_all)

            # Paint Omega_m
            if paint_om:
                fig = plt.figure(figsize=(20,20))
                for i in range(3):
                    index = self.ylabel_all == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_train_param_all[index,0],
                                marker=self.markers[i],
                                s=s, cmap='Spectral');
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_om_train_ext.png", dpi=200,bbox_inches='tight')
                plt.show()

            # Paint sigma_8
            if paint_om:
                fig = plt.figure(figsize=(20,20))
                for i in range(3):
                    index = self.ylabel_all == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_train_param_all[index,1],
                                marker=self.markers[i],
                                s=s, cmap='Spectral');
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_sg_train_ext.png", dpi=200,bbox_inches='tight')
                plt.show()

            fig = plt.figure(figsize=(20,20))
            plt.scatter(umap_pred[:, 0], umap_pred[:, 1],
                        c=self.ylabel_all, s=s, cmap='Spectral');
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if fname is not None:
                plt.savefig(f"img/umap_{fname}_train_ext.png", dpi=200,bbox_inches='tight')
            plt.show()
            if connectivity:
                fig = plt.figure(figsize=(20,20))
                umap.plot.connectivity(mapper, labels=self.ylabel_all, show_points=True)
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.show()

            if return_data:
                return umap_pred, self.ylabel_all

        else:
            if dens_map:
                embedding = umap.UMAP(
                    densmap=True, dens_lambda=dens_lambda,
                    random_state=random_state)
            else:
                embedding = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=n_components,
                                      random_state=random_state)
            mapper    = embedding.fit(self.Z_test)
            umap_pred = embedding.transform(self.Z_test)

            # Paint Omega_m
            if paint_om:
                fig = plt.figure(figsize=(20,20))
                for i in range(2):
                    index = self.clabel == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_test_param[index,0],
                                marker=self.markers[i],
                                s=s, cmap='Spectral', zorder=self.zorders[i]);
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_om_test.png", dpi=200,bbox_inches='tight')
                plt.show()

            # Paint sigma_8
            if paint_sg:
                fig = plt.figure(figsize=(20,20))
                for i in range(2):
                    index = self.clabel == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_test_param[index,1],
                                marker=self.markers[i],
                                s=s, cmap='Spectral', zorder=self.zorders[i]);
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_sg_test.png", dpi=200,bbox_inches='tight')
                plt.show()

            fig = plt.figure(figsize=(20,20))
            plt.scatter(umap_pred[:, 0], umap_pred[:, 1],
                        c=self.clabel, s=s, cmap='Spectral');
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if fname is not None:
                plt.savefig(f"img/umap_{fname}_test.png", dpi=200,bbox_inches='tight')
            plt.show()
            if connectivity:
                fig = plt.figure(figsize=(20,20))
                umap.plot.connectivity(mapper, labels=self.clabel, show_points=True)
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.show()

            if return_data:
                return umap_pred, self.ylabel


            if dens_map:
                embedding = umap.UMAP(
                    densmap=True, dens_lambda=dens_lambda,
                    random_state=random_state)
            else:
                embedding = umap.UMAP(n_neighbors=n_neighbors,
                                      min_dist=min_dist,
                                      n_components=n_components,
                                      random_state=random_state)
            mapper    = embedding.fit(self.Z_train)
            umap_pred = embedding.transform(self.Z_train)

            # Paint Omega_m
            if paint_om:
                fig = plt.figure(figsize=(20,20))
                for i in range(2):
                    index = self.ylabel == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_train_param[index,0],
                                marker=self.markers[i],
                                s=s, cmap='Spectral');
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_om_train.png", dpi=200,bbox_inches='tight')
                plt.show()

            # Paint sigma_8
            if paint_sg:
                fig = plt.figure(figsize=(20,20))
                for i in range(2):
                    index = self.ylabel == self.colors[i]
                    plt.scatter(umap_pred[index, 0], umap_pred[index, 1],
                                c=self.y_train_param[index,1],
                                marker=self.markers[i],
                                s=s, cmap='Spectral');
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if fname is not None:
                    plt.savefig(f"img/umap_{fname}_sg_train.png", dpi=200,bbox_inches='tight')
                plt.show()

            fig = plt.figure(figsize=(20,20))
            plt.scatter(umap_pred[:, 0], umap_pred[:, 1],
                        c=self.ylabel, s=s, cmap='Spectral');
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if fname is not None:
                plt.savefig(f"img/umap_{fname}_train.png", dpi=200,bbox_inches='tight')
            plt.show()
            if connectivity:
                fig = plt.figure(figsize=(20,20))
                umap.plot.connectivity(mapper, labels=self.ylabel, show_points=True)
                ax = plt.gca()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.show()

        if return_data:
            return umap_pred, self.ylabel



    def return_umap(self,n_neighbors=40,min_dist=0.1, n_components=2,
                    dens_map=False, dens_lambda=2.0,
                    random_state=3,
                    which_data='test',
                    ext_sim=False,
                    ):

        if dens_map:
            embedding = umap.UMAP(
                densmap=True, dens_lambda=dens_lambda,
                random_state=random_state)
        else:
            embedding = umap.UMAP(n_neighbors=n_neighbors,
                                  min_dist=min_dist,
                                  n_components=n_components,
                                  random_state=random_state)

        if which_data == 'test' and ext_sim:
            data = self.Z_test_all
            label = [self.clabel_all, self.y_test_param_all]
        elif which_data == 'train' and ext_sim:
            data = self.Z_train_all
            label = [self.ylabel_all, self.y_train_param_all]
        elif which_data == 'test' and (not ext_sim):
            data = self.Z_test
            label = [self.clabel, self.y_test_param]
        elif which_data == 'train' and (not ext_sim):
            data = self.Z_train
            label = [self.ylabel, self.y_train_param]

        mapper    = embedding.fit(data)
        umap_pred = embedding.transform(data)
        return mapper, data, umap_pred, label
