import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *

field = 'T';
sim   = 'TNG'
norm = True
mono = True
h    = 0.5 #1
average=False
fname = "search/{}_{}_mono_{}_norm_{}_h_{}_img".format(sim, field,
                                                       int(mono),int(norm),h)
mist = MIST(sim=sim,field=field, extended_L=False, normalization=norm,
            monopole=mono, average=average, data_type='image')
mist.load_models(fname=fname, which_machine='vib_cnn',z_dim=200, hidden=h,
                 dr=0.5)
mist.make_plots(fname="cnn_{}_{}_0.5".format(field,sim),show_plot=False, data_return=False,save_plot=True)


field = 'T';
sim   = 'TNG'
norm = True
mono = True
h    = 1
average=False
fname = "search/{}_{}_mono_{}_norm_{}_h_{}_img".format(sim, field,
                                                       int(mono),int(norm),h)
mist = MIST(sim=sim,field=field, extended_L=False, normalization=norm,
            monopole=mono, average=average, data_type='image')
mist.load_models(fname=fname, which_machine='vib_cnn',z_dim=200, hidden=h,
                 dr=0.5)
mist.make_plots(fname="cnn_{}_{}_1".format(field,sim),show_plot=False, data_return=False,save_plot=True)


field = 'T';
sim   = 'TNG'
norm = True
mono = True
h    = 0.25
average=False
fname = "search/{}_{}_mono_{}_norm_{}_h_{}_img".format(sim, field,
                                                       int(mono),int(norm),h)
mist = MIST(sim=sim,field=field, extended_L=False, normalization=norm,
            monopole=mono, average=average, data_type='image')
mist.load_models(fname=fname, which_machine='vib_cnn',z_dim=200, hidden=h,
                 dr=0.5)
mist.make_plots(fname="cnn_{}_{}_0.25".format(field,sim),show_plot=False, data_return=False,save_plot=True)
