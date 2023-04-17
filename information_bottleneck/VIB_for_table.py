import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *


# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



fields = ['Mtot','Mgas','Mstar','HI','ne','Vcdm','Z','T']
sims   = ["TNG","SIMBA","ASTRID","GADGET","RAMSES"]


##  For long epoch!
## 1. TNG+SIMBA
## 2. TNG+SIMBA+ASTRID
## 3. GADGET+RAMSES
## 4. GADGET+TNG
if False:
    field  = 'Mtot'
    sims   = [['TNG', 'SIMBA'],     ["TNG", "SIMBA", "ASTRID"], 
              ["GADGET", "RAMSES"], ["GADGET", "TNG"]]
    fnames = ["TNG+SIMBA", "TNG+SIMBA+ASTRID", "GADGET+RAMSES", "GADGET+TNG"]
    for i in range(len(fnames)):
        print("{} starts!".format(fnames[i]))
        mist = MIST(sim=sims[i],field=field)
        mist.train(fname=fnames[i],epochs=30000,verbose=False)
        mist.print_score()
        mist.make_plots(fname=fnames[i])
        print("----------------------------------------------------")
        print("\n")
        print("\n")






## 1. TNG+SIMBA+ASTRID for field
if False:
    fields  = ['Mtot','Mgas','Mstar','HI','ne','Vcdm','Z','T']
    sims    = ["TNG", "SIMBA", "ASTRID"]
    fnames  = "TNG+SIMBA+ASTRID"
    for i in range(len(fields)):
        field = fields[i]
        fname = fnames+"_{}".format(field)
        sim   = sims

        print("{} starts!".format(fname))
        mist = MIST(sim=sim,field=field)
        mist.train(fname=fname,epochs=1000,verbose=False)
        mist.print_score()
        mist.make_plots(fname=fname)
        print("----------------------------------------------------")
        print("\n")
        print("\n")


## 1. TNG+SIMBA for  HI
if False:
    fields  = ['HI']
    sims    = ["TNG", "SIMBA"]
    fnames  = "TNG+SIMBA"
    for i in range(len(fields)):
        field = fields[i]
        fname = fnames+"_{}".format(field)
        sim   = sims
        print("{} starts!".format(fname))
        mist = MIST(sim=sim,field=field)
        mist.train(fname=fname,epochs=1000,verbose=False)
        mist.print_score()
        mist.make_plots(fname=fname)
        print("----------------------------------------------------")
        print("\n")
        print("\n")




## 1. TNG+SIMBA extended l for field 
if False:
    fields  = ['Mtot','Mgas','Mstar','HI','ne','Vcdm','Z','T']
    sims    = ["TNG", "SIMBA"]
    fnames  = "TNG+SIMBA"
    for i in range(len(fields)):
        field = fields[i]
        fname = fnames+"_{}_l_10".format(field)
        sim   = sims

        print("{} starts!".format(fname))
        mist = MIST(sim=sim,field=field)
        mist.train(fname=fname,epochs=1000,verbose=False)
        mist.print_score()
        mist.make_plots(fname=fname)
        print("----------------------------------------------------")
        print("\n")
        print("\n")


## For VIB
## 1. TNG+SIMBA
## 2. TNG+SIMBA+ASTRID
## 3. GADGET+RAMSES
## 4. GADGET+TNG
if False:
    field  = 'Mtot'
    sims   = [['TNG', 'SIMBA'],     ["TNG", "SIMBA", "ASTRID"], 
              ["GADGET", "RAMSES"], ["GADGET", "TNG"]]
    fnames = ["TNG+SIMBA_vib", "TNG+SIMBA+ASTRID_vib", "GADGET+RAMSES_vib", "GADGET+TNG_vib"]
    for i in range(len(fnames)):
        print("{} starts!".format(fnames[i]))
        mist = MIST(sim=sims[i],field=field)
        mist.train(fname=fnames[i],epochs=30000,verbose=False,
                   which_machine="vib")
        mist.print_score()
        mist.make_plots(fname=fnames[i])
        print("----------------------------------------------------")
        print("\n")
        print("\n")




## For VIB, search beta & gamma
## 1. TNG+SIMBA
if False: 
    field  = 'Mtot'
    sim    = ['TNG', 'SIMBA']
    betas  = [0.01, 0.1, 1., 10, 100,]
    gammas = [0.01, 0.1, 1., 10, 100]
    #for beta in betas:
    #    for gamma in gammas:

    beta  = 0.1 #[0.01, 0.1, 1., 10, 100,]
    gamma = 1.0 #[0.01, 0.1, 1., 10, 100]

    fname = "search/TNG+SIMBA_beta_{}_gamma_{}".format(beta, gamma)
    print("{} starts!".format(fname))
    mist = MIST(sim=sim,field=field)
    mist.train(fname=fname, epochs=10000,
               z_dim=1000,
               verbose=False,
               which_machine="vib+cls",beta=beta, gamma=gamma)
    mist.print_score()
    mist.make_plots(fname=fname)
    print("----------------------------------------------------")
    print("\n")
    print("\n")




## Some productions
if False: 
    #field  = 'Mtot'
    field  = 'HI'
    #field  = 'Mgas'
    #field  = 'T'
    #sim    = ['TNG', 'GADGET']
    sim    = ['TNG', 'SIMBA']
    #sim    = ['TNG', 'SIMBA', "ASTRID"]
    betas  = [0.01, 0.1, 1., 10]
    gammas = [100., 1000, 1000]
    #for beta in betas:
    #    for gamma in gammas:

    #betas  = [0.01,]# 0.01, 0.1, 1., 10, 100]
    #gammas = [0.,]# 0.01, 0.1, 1., 10, 100]

    for beta in betas:
        for gamma in gammas:
            fname = "search/TNG+SIMBA_{}_beta_{}_gamma_{}".format(field, beta, gamma)
            #fname = "search/TNG+SIMBA+ASTRID_{}_beta_{}_gamma_{}".format(field, beta, gamma)
            print("{} starts!".format(fname))
            mist = MIST(sim=sim,field=field)
            mist.train(fname=fname, epochs=5000,
                       z_dim=1000,
                       verbose=True, learning_rate=1e-3, decay_rate=0.97,
                       which_machine="vib+cls", beta=beta, gamma=gamma)
            mist.get_score()
            mist.make_plots(fname=fname)
            print("----------------------------------------------------")
            print("\n")
            print("\n")





## For grid search on
### 1. Monopole
### 2. NN structure
if False: 
    #field  = 'Mtot'
    field  = 'HI'
    sim    = ['TNG', 'SIMBA']
    betas  = 0 
    gammas = 0 
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =1e-2;
    z_dim = 200; lr=1e-3; decay_rate=0.97

    #betas  = [0.01,]# 0.01, 0.1, 1., 10, 100]
    #gammas = [0.,]# 0.01, 0.1, 1., 10, 100]
    monos = [True, False]
    norms = [True]
    hs    = [0.5]#[0.5, 1, 2]
    average = True
    success = False

    for mono in monos:
        for norm in norms:
            for h in hs:
                fname = "search/TNG+SIMBA_{}_mono_{}_norm_{}_h_{}".format(
                    field,
                    int(mono),
                    int(norm),
                    h)
                print("{} starts!".format(fname))
                mist    = MIST(sim=sim, field=field, batch_size=200, 
                               normalization=norm, monopole=mono,
                               average=average,device=device)
                while ~success:
                    success = mist.train(epochs=3000,
                                         verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                         which_machine="vib+cls_a", 
                                         beta=beta, gamma=gamma,
                                         z_dim=z_dim, hidden=h, dr=dr,
                                         fname=fname,
                                         save_plot=True, save_model=True,
                                        )
                    if success:
                        mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                                mist.get_score(bias_test=False, _print=True)
                mist.make_plots(fname=fname)
                print("----------------------------------------------------")
                print("\n")
                print("\n")



## Extended ones
if False: 
    #field  = 'Mtot'
    field  = 'HI'
    sim    = ['TNG', 'SIMBA']
    betas  = 0 
    gammas = 0 
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =0;
    z_dim = 1000; lr=1e-3; decay_rate=0.97

    #betas  = [0.01,]# 0.01, 0.1, 1., 10, 100]
    #gammas = [0.,]# 0.01, 0.1, 1., 10, 100]
    monos       = [True] #[True, False]
    norms       = [True] #[True, False]
    hs          = [1] #[0.5, 1, 2]
    extended_L  = 10
    extended_dn = 2 #2
    average     = False

    for mono in monos:
        for norm in norms:
            for h in hs:
                fname = \
                        "search/TNG+SIMBA_{}_mono_{}_norm_{}_h_{}_L_10_dn_2_z_200_j1".format(
                            field,
                            int(mono),
                            int(norm),
                            h)
                print("{} starts!".format(fname))
                mist    = MIST(sim=sim, field=field, batch_size=100, 
                               extended_L=extended_L, extended_dn=extended_dn,
                               normalization=norm, monopole=mono, 
                               average=average, device=device)
                success = mist.train(epochs=3000,
                                     verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                     which_machine="vib+cls_a", 
                                     beta=beta, gamma=gamma,
                                     z_dim=z_dim, hidden=h, dr=dr,
                                     fname=fname,
                                     save_plot=True, save_model=True,
                                    )
                mist.make_plots(fname=fname)
                print("----------------------------------------------------")
                print("\n")
                print("\n")







if False: 
    field  = 'Mtot'
    #field  = 'T'
    sim    = 'TNG'#, 'SIMBA']
    #sim    = 'SIMBA'#, 'SIMBA']
    betas  = 0 
    gammas = 0 
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =1e-2;
    z_dim = 1000; lr=1e-3; decay_rate=0.97

    #betas  = [0.01,]# 0.01, 0.1, 1., 10, 100]
    #gammas = [0.,]# 0.01, 0.1, 1., 10, 100]
    monos = [True, False]
    norms = [True, False]

    monos = [True]
    norms = [True]
    hs    = [2] #[0.5, 1, 2]
    #average = False
    average = True


    for mono in monos:
        for norm in norms:
            for h in hs:
                fname = "search/{}_{}_mono_{}_norm_{}_h_{}_new_".format(
                    sim,
                    field,
                    int(mono),
                    int(norm),
                    h)
                print("{} starts!".format(fname))
                mist    = MIST(sim=sim, field=field, batch_size=100, 
                               normalization=norm, monopole=mono,
                               average=average, data_type='wph', device=device,
                               L=4, dn=0,
                              projection=True)
                success = mist.train(epochs=3000,
                                     verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                     which_machine="vib", 
                                     beta=beta, gamma=gamma,
                                     z_dim=z_dim, hidden=h, dr=dr,
                                     fname=fname,
                                     save_plot=True, save_model=True,
                                    )
                if success:
                    mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                            mist.get_score(bias_test=False, _print=True)
                else:
                    pass
                mist.make_plots(fname=fname)
                print("----------------------------------------------------")
                print("\n")
                print("\n")




## loop over fields
if False: 
    fields = ['Mtot', 'T', 'Mgas','Mstar','HI','ne','Vcdm','Z']
    sim    = 'TNG'#, 'SIMBA']
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =1e-2;
    z_dim = 200; lr=1e-3; decay_rate=None

    mono       = True
    norm       = True
    h          = 1
    average    = False
    projection = True


    for field in fields:
        fname = "search/{}_{}_h_{}_l_4_dn_0".format(
            sim,
            field,
            int(mono),
            int(norm),
            h)
        print("{} starts!".format(fname))
        mist    = MIST(sim=sim, field=field, batch_size=100, 
                       normalization=norm, monopole=mono,
                       average=average, data_type='wph', device=device,
                       L=4, dn=0,
                       projection=projection)
        success = mist.train(epochs=3000,
                             verbose=True, learning_rate=lr, decay_rate=decay_rate,
                             which_machine="vib", 
                             beta=beta, gamma=gamma,
                             z_dim=z_dim, hidden=h, dr=dr,
                             fname=fname,
                             save_plot=True, save_model=True,
                            )
        if success:
            mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                    mist.get_score(bias_test=False, _print=True)
        else:
            pass
        mist.make_plots(fname=fname)
        print("----------------------------------------------------")
        print("\n")
        print("\n")





## loop over combinations of fields
if False: 
    fields = [['T', 'Mgas'],['T','HI'],['T','ne'],['T','Z'], ['T','HI','ne']]
    sim    = 'TNG'#, 'SIMBA']
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =1e-2;
    z_dim = 200; lr=1e-3; decay_rate=None

    mono       = True
    norm       = True
    h          = 1
    average    = False
    projection = True


    for field in fields:
        field_name = field[0]
        for fiel in field[1:]:
            field_name += '_'
            field_name += fiel
        fname = "search/{}_{}".format(
            sim,
            field_name,
            )
        print("{} starts!".format(fname))
        mist    = MIST(sim=sim, field=field, batch_size=100, 
                       normalization=norm, monopole=mono,
                       average=average, data_type='wph', device=device,
                       L=4, dn=0,
                       projection=projection)
        success = mist.train(epochs=3000,
                             verbose=True, learning_rate=lr, decay_rate=decay_rate,
                             which_machine="vib", 
                             beta=beta, gamma=gamma,
                             z_dim=z_dim, hidden=h, dr=dr,
                             fname=fname,
                             save_plot=True, save_model=True,
                            )
        if success:
            mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                    mist.get_score(bias_test=False, _print=True)
        else:
            pass
        mist.make_plots(fname=fname)
        print("----------------------------------------------------")
        print("\n")
        print("\n")


## explore hyperparameter of the loss function
if True: 
    field = 'HI'#'ne'#['HI', 'ne']
    sim    = ['TNG', 'SIMBA']
    #for beta in betas:
    #    for gamma in gammas:
    z_dim = 400; lr=1e-3; decay_rate=None
    fe = 0.3; fd=0.6; dr=0.1
    batch_size= 512

    gammas  = [0.001, 0.01, 0.1,1]
    betas = [0.1, 1, 10, 100]

    mono       = True
    norm       = True
    average    = False
    projection = False


    for beta in betas:
        for gamma in gammas:
            fname = "search/TNG_SIMBA_{}_b_{}_g_{}".format(
                field,
                beta,
                gamma,
                )
            print("{} starts!".format(fname))
            mist    = MIST(sim=sim, field=field, batch_size=batch_size, 
                           normalization=norm, monopole=mono,
                           average=average, data_type='wph', device=device,
                           L=4, dn=0,
                           projection=projection)
            success = mist.train(epochs=8000,
                                 verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                 which_machine="vib+cls", 
                                 beta=beta, gamma=gamma,
                                 z_dim=z_dim, hidden1=fe, hidden2=fd, dr=dr,
                                 patience=50,
                                 fname=fname,
                                 save_plot=True, save_model=True,
                                )
            if success:
                mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                        mist.get_score(bias_test=False, _print=True)
            else:
                pass
            mist.make_plots(fname=fname)
            print("----------------------------------------------------")
            print("\n")
            print("\n")


## explore hyperparameter of the loss function for multiple inputs
if False: 
    field = ['HI', 'ne']
    sim    = ['TNG', 'SIMBA']
    #for beta in betas:
    #    for gamma in gammas:
    z_dim = 400; lr=1e-3; decay_rate=None
    fe = 0.3; fd=0.6; dr=0.1

    gammas  = [0.01, 0.1,1]
    betas = [1, 10, 100]

    mono       = True
    norm       = True
    average    = False
    projection = True

    field_name = field[0]
    for fiel in field[1:]:
        field_name += '_'
        field_name += fiel

    for beta in betas:
        for gamma in gammas:
            fname = "search/TNG_SIMBA_{}_b_{}_g_{}".format(
                field_name,
                beta,
                gamma,
                )
            print("{} starts!".format(fname))
            mist    = MIST(sim=sim, field=field, batch_size=100, 
                           normalization=norm, monopole=mono,
                           average=average, data_type='wph', device=device,
                           L=4, dn=0,
                           projection=projection)
            success = mist.train(epochs=3000,
                                 verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                 which_machine="vib+cls", 
                                 beta=beta, gamma=gamma,
                                 z_dim=z_dim, hidden1=fe, hidden2=fd, dr=dr,
                                 fname=fname,
                                 save_plot=True, save_model=True,
                                )
            if success:
                mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                        mist.get_score(bias_test=False, _print=True)
            else:
                pass
            mist.make_plots(fname=fname)
            print("----------------------------------------------------")
            print("\n")
            print("\n")











#### IMAGE
if False: 
    #field  = 'Mtot'
    field  = 'T'
    sim    = 'TNG'#, 'SIMBA']
    #sim    = 'SIMBA'#, 'SIMBA']
    betas  = 0 
    gammas = 0 
    #for beta in betas:
    #    for gamma in gammas:
    dr = 0.5; beta=0; gamma =1e-2;
    z_dim = 1000; lr=1e-3; decay_rate=0.97

    #betas  = [0.01,]# 0.01, 0.1, 1., 10, 100]
    #gammas = [0.,]# 0.01, 0.1, 1., 10, 100]
    monos = [True, False]
    norms = [True, False]
    hs = [1/4]
    monos = [True]
    norms = [True]
    average = False


    for mono in monos:
        for norm in norms:
            for h in hs:
                fname = "search/{}_{}_mono_{}_norm_{}_h_{}_img".format(
                    sim,
                    field,
                    int(mono),
                    int(norm),
                    h)
                print("{} starts!".format(fname))
                mist    = MIST(sim=sim, field=field, batch_size=200, 
                               normalization=norm, monopole=mono,
                               average=average, data_type='image', device=device)
                success = mist.train(epochs=3000,
                                     verbose=True, learning_rate=lr, decay_rate=decay_rate,
                                     which_machine="vib_cnn", 
                                     beta=beta, gamma=gamma,
                                     z_dim=z_dim, hidden=h, dr=dr,
                                     fname=fname,
                                     save_plot=True, save_model=True,
                                    )
                if success:
                    mse_om, mse_sig, chi2_om, chi2_sig, auc =\
                            mist.get_score(bias_test=False, _print=True)
                else:
                    pass
                mist.make_plots(fname=fname)
                print("----------------------------------------------------")
                print("\n")
                print("\n")



