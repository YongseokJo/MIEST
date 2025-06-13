from functools import partial
import numpy as np
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
from mist_utils import *
#import networks
#import ray_utils
#from networks import *
#from ray_utils import *




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




class RayTrainer:
    def __init__(self, params):
        # Some infos
        self.machine = {'vib':VIB, 'vib+cls':VIB}
        self.is_cls  = ['vib+cls']
        self.loss    = {}
        self.names_loss = ['total_loss', 'j0_loss', 'j1_loss', 'ce0_loss',
                      'ce1_loss']

        # Arguments
        self.which_machine = params["which_machine"]
        self.train_loader  = params["trainset"]
        self.val_dataset   = params["validset"]
        self.num_sim       = params["num_sim"]
        self.max_epoch     = params["max_epoch"]
        self.input_shape   = params["input_shape"]
        self.output_shape  = params["output_shape"]
        self.patience      = params["patience"]

        self.num_loss      = 5 if self.which_machine in self.is_cls else 4

        # Early Stopper
        self.log_container      = Log([],batch_size=self.train_loader.batch_size)
        self.early_stopping     = EarlyStopping(None, tol=None, patience=self.patience)
        self.callback_container = CallbackContainer([self.log_container,
                                                     self.early_stopping])
        self.callback_container.set_trainer(self)


    def run(self, config, checkpoint_dir=None):
        # Init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_epoch = 0
        save_plot = False

        # Network
        self.vib = self.machine[self.which_machine](input_shape=self.input_shape,
                                     output_shape=self.output_shape,
                                     z_dim=config['z_dim'], fe=config['fe'],
                                     fd=config['fd'], dr=config['dr'])
        if torch.cuda.device_count() > 1:
            self.vib = nn.DataParallel(self.vib)
        self.vib.to(self.device)
        self.optimizer = torch.optim.Adam(self.vib.parameters(),lr=config['lr'])

        if self.which_machine in self.is_cls:
            self.cls = classifier(config['z_dim'], num_models=num_sim)
            if torch.cuda.device_count() > 1:
                self.cls = nn.DataParallel(self.cls)
            self.cls.to(self.device)
            self.cls_optimizer = torch.optim.Adam(self.cls.parameters(), lr=1e-3)


        # Check point
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint_vib"))
            self.vig.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

            if self.which_machine in self.is_cls:
                model_state, optimizer_state = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint_cls"))
                self.cls.load_state_dict(model_state)
                cls_optimizer.load_state_dict(optimizer_state)

        self.callback_container.on_train_begin()

        # Start training
        for epoch in range(self.max_epoch):
            self.callback_container.on_epoch_begin(epoch)

            self.batch_accuracy = 0

            # Batch start
            for batch_idx, (X,y) in enumerate(self.train_loader):
                self.callback_container.on_batch_begin(batch_idx)
                X       = X.to(self.device)
                y       = y.float().to(self.device)
                if self.which_machine in self.is_cls:
                    self.y_cls = y[:,-self.num_sim:]
                else:
                    self.y_cls = None
                self.y_param = y[:,:-self.num_sim]

                # VIB
                self.losses, self.pred =\
                        self.train_vib(X, self.y_param, config['beta'], config['gamma'],
                                       y_cls=self.y_cls)
                if torch.isnan(self.losses[0]) or torch.isinf(self.losses[0]):
                    return self.vib, self.cls, False

                # CLS
                if self.which_machine in self.is_cls:
                    cls_loss = self.train_cls(X,self.y_cls)

                self.callback_container.on_batch_end(batch_idx)

            # batch ended
            self.callback_container.on_epoch_end(epoch,
                                                 logs=self.log_container.history)

            """
            # Save models
            torch.save(self.vib.state_dict(),"./model/tmp/{}_{}_vib.pt".format(self.key,epoch))
            if self.which_machine in self.mach_vib_cls:
                torch.save(self.cls.state_dict(),"./model/tmp/{}_{}_cls.pt".format(self.key,epoch))

            new_best_epoch = self.log_container.history['best_epoch']

            # Delete models which do not fall into the convergence region
            if best_epoch != new_best_epoch:
                for i in range(best_epoch, new_best_epoch):
                    fmodel = "./model/tmp/{}_{}".format(self.key, i)
                    if os.path.exists(fmodel+'_vib.pt'):
                        os.remove(fmodel+'_vib.pt')
                    if self.which_machine in self.mach_vib_cls:
                        if os.path.exists(fmodel+'_cls.pt'):
                            os.remove(fmodel+'_cls.pt')
                best_epoch = new_best_epoch
                """


            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint_vib")
                torch.save((self.vib.state_dict(), self.optimizer.state_dict()),path)
                if self.which_machine in self.is_cls:
                    path = os.path.join(checkpoint_dir, "checkpoint_cls")
                    torch.save((self.cls.state_dict(), self.cls_optimizer.state_dict()),path)

            tune.report(loss=self.log_container.history['val_loss'][-1],
                        accuracy=self.log_container.history['val_accuray'][0])


            #if self.log_container.history['stop_training']:
            #    break

        # epoch ended 
        self.callback_container.on_train_end()
        print('Finished Training')

        """
        # Load the best model
        fmodel = "./model/tmp/{}_{}".format(self.key, best_epoch)
        self.vib.load_state_dict(torch.load(fmodel+'_vib.pt'))
        if self.which_machine in self.mach_vib_cls:
            self.cls.load_state_dict(torch.load(fmodel+'_cls.pt'))

        # Delete all tmp models
        for i in range(best_epoch, epoch+1):
            fmodel = "./model/tmp/{}_{}".format(self.key, i)
            if os.path.exists(fmodel+'_vib.pt'):
                os.remove(fmodel+'_vib.pt')
            if self.which_machine in self.mach_vib_cls:
                if os.path.exists(fmodel+'_cls.pt'):
                    os.remove(fmodel+'_cls.pt')
                    """


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
        if self.which_machine in self.is_cls:
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
        self.cls_optimizer.step()
        self.cls.eval()
        return cls_loss


    def train_vib(self, X, y_true, beta, gamma, y_cls=None):
        self.vib.train()
        self.vib.zero_grad()
        y_pred, y_sigma = self.vib(X)
        _mu, _std       = self.vib.get_mu_std()
        if self.which_machine in self.is_cls:
            cls_pred        = self.cls(_mu.detach()) ## This is for score
            losses = vib_cls_loss(y_pred, y_sigma, y_true, y_cls, cls_pred.detach(), _mu, _std,
                                                beta=beta, gamma=gamma)
        else:
            losses = vib_loss(y_pred, y_sigma, y_true, _mu, _std, gamma=gamma)
            cls_pred        =  None 
        losses[0].backward()
        self.optimizer.step()
        return losses, (y_pred, y_sigma, cls_pred)







def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5, params=None):
    trainer = RayTrainer(params=params)
    config = {
        'z_dim': tune.randint(50,2000),
        'fe'   : tune.uniform(0.001,1.),
        'fd'   : tune.uniform(0.001,1.),
        'dr'   : tune.uniform(0.001,0.9),
        'lr'   : tune.loguniform(1e-5,1e-1),
        'beta' : tune.loguniform(1e-2,1e+3),
        'gamma': tune.loguniform(1e-4,1e+2),
    }

    scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=params['max_epoch'],
                grace_period=1,
                reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        #, "accuracy"
        metric_columns=["loss", 'accuracy', "training_iteration"])
    result = tune.run(
        trainer.run, # partial
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # Test part
    """
    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    #best_trained_model.to(device)

    #best_checkpoint_dir = best_trial.checkpoint.value
    #model_state, optimizer_state = torch.load(os.path.join(
    #    best_checkpoint_dir, "checkpoint"))
    #best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(best_trained_model, device)
    #print("Best trial test set accuracy: {}".format(test_acc))
    """


if __name__ == '__main__':
    field = 'HI';
    sim   = 'TNG' #['TNG', 'SIMBA']
    monopole = True;
    projection=True
    L = 4; dn = 0


    fname = "TNG_{}_ray_test".format(field)
    mist = MIST(sim=sim,field=field, normalization=True, monopole=monopole,
                average=False, data_type='wph',
                L=L, dn=dn, projection=projection,batch_size=128)
    params =  {
        'which_machine':'vib',
        'trainset'     : mist.train_loader,
        'validset'     : mist.val_dataset,
        'num_sim'      : 1,
        'max_epoch'    : 3000,
        'input_shape'  : mist.input.shape[1],
        'output_shape' : mist.output.shape[1],
        'patience'     : 50,
    }
    #ray.init(runtime_env={"py_modules": [networks, ray_utils]})
    main(num_samples=10, gpus_per_trial=0.5, params=params)


