import time
import datetime
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any
import warnings
import sklearn
import math
import torch


class Callback:

    def __init__(self):
        pass

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


@dataclass
class CallbackContainer:

    callbacks: List[Callback] = field(default_factory=list)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)


class EarlyStopping(Callback):
    """EarlyStopping callback to exit the training loop if early_stopping_metric
    does not improve by a certain amount for a certain
    number of epochs.

    Parameters
    ---------
    early_stopping_metric : str
        Early stopping metric name
    is_maximize : bool
        Whether to maximize or not early_stopping_metric
    tol : float
        minimum change in monitored value to qualify as improvement.
        This number should be positive.
    patience : integer
        number of epochs to wait for improvement before terminating.
        the counter be reset after each improvement

    """
    def __init__(self, early_stopping_metric:str, tol:float, patience:int):
        super().__init__()
        self.early_stopping_metric = early_stopping_metric
        self.tol           = tol
        self.patience      = patience
        self.best_epoch    = 0
        self.stopped_epoch = 0
        self.wait          = 0
        self.best_weights  = None
        self.best_acc      = np.inf
        self.best_loss     = np.inf
        self.best_val_acc  = np.array([np.inf, np.inf])
        self.best_val_loss = np.inf
        self.best_auc      = np.inf
        self.best_val_auc  = np.inf
        self.cnvg_penalty  = 1

    def on_epoch_end(self, epoch, logs=None):
        is_improved = False
        is_converged = True

        current_loss = logs.get("j0_loss")
        accuracy     = logs.get("accuracy")
        val_accuray  = logs.get('val_accuray')
        val_loss     = logs.get('val_loss')
        cnvg_tail    = logs.get('cnvg_tail')
        cnvg_length = \
                self.patience if self.patience >= epoch-cnvg_tail+1 else epoch-cnvg_tail+1
        if current_loss is None:
            return

        def is_best(best, now, tol=None, use_convergence=False):
            nonlocal is_improved, is_converged
            # is converged?
            if use_convergence:
                mean = np.mean(now[-cnvg_length:])
                dev = np.abs(mean-np.array(now[-cnvg_length:]))
                dev = np.mean(dev) ## MAE
                criteria =\
                        np.abs(mean*0.10/4) if use_convergence == True else use_convergence
                #print(cnvg_length, cnvg_tail,dev,criteria)
                if dev > criteria: # think it's very generous ~ 15%
                    # Not converged, gotta go further,
                    is_converged = False
                    logs['cnvg_tail'] += self.cnvg_penalty
                    self.cnvg_penalty += 1
                    # lower bound for tail according to patience
                    if self.patience > (epoch+1)-logs['cnvg_tail']+1:
                        logs['cnvg_tail'] = \
                                np.maximum(epoch+1-self.patience+1, 0)
                        self.cnvg_penalty = 1
                else:
                    # reset the penalty
                    self.cnvg_penalty = 1
                now = now[-1]

            # is improved?
            if now < best:
                is_improved = True
                return now
            else:
                return best



        self.best_val_loss   = is_best(self.best_val_loss, val_loss[-1])
        self.best_val_acc[0] = is_best(self.best_val_acc[0], val_accuray[0])
        self.best_val_acc[1] = is_best(self.best_val_acc[1], val_accuray[1])
        self.best_acc        = is_best(self.best_acc, accuracy[-1])#,use_convergence=True)
        self.best_loss       = is_best(self.best_loss, current_loss, use_convergence=True)

        if self.trainer.which_machine in self.trainer.is_cls:
            auc               = np.abs(np.array(logs.get('auc_score'))-0.5)[-1]
            val_auc           = np.abs(np.array(logs.get('auc_score_val'))-0.5)
            self.best_auc     = is_best(self.best_auc, auc)
            self.best_val_auc = is_best(self.best_val_auc, val_auc, use_convergence=0.02)


        if (is_improved) or (not is_converged):
            self.wait = 1
        else:
            # early stop?
            if self.wait >= self.patience:
                self.stopped_epoch   = epoch
                logs["stop_training"] = True
            # let's  wait
            self.wait += 1
        # get the best epoch within the convergence region
        cnvg_val_loss         = np.array(val_loss[-cnvg_length:])
        self.best_epoch_loss  = cnvg_val_loss.min()
        best_val_index        = np.argmin(cnvg_val_loss)
        best_index            = (epoch+1) - cnvg_length + best_val_index \
                if epoch+1 >= cnvg_length else best_val_index
        #print(cnvg_val_loss.min(), val_loss[best_index])
        self.best_epoch    = best_index
        logs['best_epoch'] = best_index

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occurred at epoch {self.stopped_epoch}"
            msg += (
                f" with best_epoch = {self.best_epoch} and "
                + f"best_val_loss = {round(self.best_epoch_loss, 5)}"
            )
            print(msg)
        else:
            msg = (
                f"Stop training"
                + f" with best_epoch = {self.best_epoch} and "
                + f"best_val_loss = {round(self.best_epoch_loss, 5)}"
            )
            print(msg)
        wrn_msg = "Best weights from best epoch are automatically used!"
        warnings.warn(wrn_msg)


@dataclass
class Log(Callback):

    metrics_names:list
    batch_size:int
    verbose: int = 1

    def __post_init__(self):
        super().__init__()
        self.samples_seen = 0.0
        self.total_time = 0.0
        self.early_stopping = {}
        self.history = {}
        self.epoch_loss = {}

    def on_train_begin(self, logs=None):
        for name in self.trainer.names_loss:
            self.history.update({name: []})
            self.epoch_loss.update({name: 0.})
        self.history.update({"auc_score": []})
        self.history.update({"auc_score_val": []})
        self.history.update({"accuracy": []})
        self.history.update({"best_epoch": 0})
        self.history.update({"stop_training": False})
        self.history.update({"best_weights": None})
        self.history.update({"val_accuray": None})
        self.history.update({"val_loss": []})
        self.history.update({"cnvg_tail": 0})
        self.history.update({name: [] for name in self.metrics_names})

    def on_epoch_begin(self, epoch, logs=None):
        self.samples_seen   = 0.0
        self.epoch_accuracy = 0.0
        for name in self.trainer.names_loss:
            self.epoch_loss[name] = 0.0

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.trainer.num_loss):
            name = self.trainer.names_loss[i]
            self.history[name].append(self.epoch_loss[name])
        #for metric_name, metric_value in self.epoch_metrics.items():
        #    self.history[metric_name].append(metric_value)
        #self.early_stopping["early_stopping_metric"] =\
        #        self.epoch_loss
                #self.early_stopping_metric

        if self.trainer.valid_loader is not None:
            X_val, y_val   = self.trainer.valid_loader.dataset.tensors
            index          = self.trainer.valid_loader.sampler.indices
            X_val, y_val   = X_val[index,:], y_val[index,:]
            y_param        = y_val[:,:-self.trainer.num_sim].cpu().detach().numpy()
            y_pred, _      = self.trainer.vib(X_val.to(self.trainer.device))
            y_pred         = y_pred.detach().cpu().numpy()
            val_accuray    = np.abs(y_param-y_pred)/y_param
            val_accuray    = val_accuray.mean(axis=0)
            val_loss       = (y_param-y_pred)**2
            val_loss       = np.sum(np.log(np.mean(val_loss,axis=0)))

            if self.trainer.which_machine in self.trainer.is_cls:
                cls_val       = y_val[:,-self.trainer.num_sim:]
                _mu_val, _    = self.trainer.vib.get_mu_std()
                cls_val_pred  = self.trainer.cls(_mu_val.detach())
        else:
            val_accuray = -1.0

        if self.trainer.which_machine in self.trainer.is_cls:
            cls_pred = self.trainer.pred[2].detach().cpu().numpy()
            cls_pred[np.isnan(cls_pred)] = -1
            self.history["auc_score"].append(sklearn.metrics.roc_auc_score(
                self.trainer.y_cls.clone().detach().cpu().numpy(),cls_pred))
            cls_val_pred = cls_val_pred.detach().cpu().numpy()
            cls_val_pred[np.isnan(cls_val_pred)] = -1
            self.history["auc_score_val"].append(sklearn.metrics.roc_auc_score(
                cls_val.clone().detach().cpu().numpy(),cls_val_pred))
        self.history["accuracy"].append(self.epoch_accuracy)
        self.history["val_accuray"] = np.array(val_accuray)
        self.history["val_loss"].append(np.array(val_loss))

        if self.verbose == 0:
            return
        if (epoch+1) % self.verbose != 0:
            return

        rnd = lambda x : round(float(x), 5)

        msg = f"epoch {epoch:<3}"
        msg += f" | {'loss':<3}: {rnd(self.history['j0_loss'][-1]):<8}"
        msg += f" | {'accuracy':<3}: {rnd(self.history['accuracy'][-1]):<8}"
        msg += f" | {'val_loss':<3}: {rnd(self.history['val_loss'][-1]):<8}"
        msg += f" | {'Om_m':<3}: {rnd(self.history['val_accuray'][0]):<8}"
        msg += f" | {'sig8':<3}: {rnd(self.history['val_accuray'][1]):<8}"
        """
        for metric_name, metric_value in self.epoch_metrics.items():
            if metric_name != "lr":
                msg += f"| {metric_name:<3}: {np.round(metric_value, 5):<8}"
                """
        print(msg)

    def on_batch_end(self, batch, logs=None):
        for i in range(self.trainer.num_loss):
            name = self.trainer.names_loss[i]
            self.epoch_loss.update(
                { name :
                 (self.samples_seen * self.epoch_loss[name] \
                  + self.batch_size *\
                  self.trainer.losses[i].cpu().detach().numpy())
                 / (self.samples_seen + self.batch_size) }
            )

        batch_accuracy = torch.sum(torch.abs(
            self.trainer.y_param - self.trainer.pred[0])/self.trainer.y_param)
        self.epoch_accuracy = (self.samples_seen * self.epoch_accuracy +\
                               batch_accuracy.cpu().detach().numpy())\
                               /(self.samples_seen + self.batch_size)

        self.samples_seen += self.batch_size

    def __getitem__(self, name):
        return self.history[name]



