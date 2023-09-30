'''Defining a class used to more easily manage data reporting on the performance of a Classifier.'''

import numpy as np
import pandas as pd
import pickle
import sklearn

# TODO: Get rid of some of the redundancy in this section. 

def check_df(df):
    '''Quick little function for checking DataFrames under construction.'''
    msg = 'plot.check_df: Column lengths are mismatched, ' + ' '.join(f'len({key})={len(val)}' for key, val in df.items())
    # assert (len(df['batch']) == len(df['loss'])) and (len(df['loss']) == len(df['label'])), msg
    
    length = len(list(df.values())[0])
    assert np.all([len(df[key]) == length for key in df.keys()]), msg


def check_type(x, t):
    pass
    # assert type(x) is not None, 'Input value is None.'
    # assert type(x) == t, f'Expected type {repr(t)}, got {repr(type(x))}.'


class Reporter():
    '''A class for storing information associated with model performance.'''

    def __init__(self, epochs=None, lr=None, bce_loss_weight=None):

        # Metrics will be added as methods are called.
        self.loss_metrics = {}
        self.acc_metrics = {}

        # Contains tuples (tn, fp, fn, tp)
        self.confusion_matrix = None

        self.active = False
        
        self.epochs = epochs
        self.lr = lr
        self.bce_loss_weight = bce_loss_weight

        # TODO: Add stuff for test losses. 

        # To be populated later...
        self.batches = None
        self.batches_per_epoch = None

    def check_active(self):
        '''Check if the Reporter is active.'''
        assert self.active, 'Reporter object must be active to add metrics.'

    def open(self):
        self.active = True

    def close(self):
        self.active = False
        # Now that logging is complete, load the number of batches which have been processed, if any. 
        if 'train_losses' in self.loss_metrics:
            self.batches = len(self.loss_metrics['train_losses'])
    
    def set_batches_per_epoch(self, batches_per_epoch):
        check_type(batches_per_epoch, int)
        self.batches_per_epoch = batches_per_epoch

    def __add(self, value:float=None, metric:str=None, group:dict=None):
        '''Add a value under the specified metric in the internal list given by 
        the 'group' keyword argument.
        
        args:
            - value: The loss or value accuracy to add to the instance. 
            - metric: The name of the metric which the value belongs to. 
            - group: The group (either self.loss_metrics or self.acc_metrics) which the metric belongs to. 
        '''
        self.check_active() # Can only add to the reporter when the instance is active. 
        check_type(value, float)
        if metric not in group:
            group[metric] = []
        group[metric].append(value)

    def add_train_loss(self, loss):
        '''Add a train_loss to the internal list.'''
        self.__add(value=loss.item(), metric='train_losses', group=self.loss_metrics)

    def add_train_acc(self, acc):
        '''Add a train_acc to the internal list.'''
        self.__add(value=acc, metric='train_accs', group=self.acc_metrics)

    def add_train_metrics(self, loss, acc):
        self.add_train_loss(loss)
        self.add_train_acc(acc)

    def add_val_loss(self, loss):
        '''Add a val_loss to the internal list.'''
        self.__add(value=loss.item(), metric='val_losses', group=self.loss_metrics)

    def add_val_acc(self, acc):
        '''Add a val_acc to the internal list.'''
        self.__add(value=acc, metric='val_accs', group=self.acc_metrics)

    def add_val_metrics(self, loss, acc):
        self.add_val_loss(loss)
        self.add_val_acc(acc)

    def add_test_loss(self, loss):
        '''Add a train_loss to the internal list.'''
        self.__add(value=loss.item(), metric='test_losses', group=self.loss_metrics)

    def add_test_acc(self, acc):
        '''Add a train_acc to the internal list.'''
        self.__add(value=acc, metric='test_accs', group=self.acc_metrics)

    def add_test_metrics(self, loss, acc):
        self.add_test_loss(loss)
        self.add_test_acc(acc)

    def get_epoch_batches(self):
        '''Get the batch numbers where each new epoch begins.'''
        assert self.batches_per_epoch is not None, 'batches_per_epoch attribute has not been set.'
        step = self.batches_per_epoch
        return [i for i in range(0, self.batches + step, step)]
    
    def add_confusion_matrix(self, tn:int, fp:int, fn:int, tp:int) -> None:
        '''Add new confusion matrix data to the internal list.'''
        self.check_active()
        self.confusion_matrix = (tn, fp, fn, tp)

    def get_confusion_matrix(self) -> tuple:
        '''Return confusion matrix stored as an attribute.'''
        assert self.confusion_matrix is not None, 'reporter.Reporter.get_confusion_matrix: No confusion matrix has been logged.'
        return self.confusion_matrix

    def _get_info_pooled(self, metrics, verbose=False):
        '''Use the information returned by the train function to construct a DataFrame for plotting loss. Pools
        the training loss over epochs.'''

        # Make sure the reporter object is no longer actively being logged to. 
        assert not self.active
        assert self.batches_per_epoch is not None, 'Reporter.__get_info_pooled: batches_per_epoch has not been set.'
        
        df = {'epoch':[], 'value':[], 'metric':[]}

        for metric, data in metrics.items():
            # First add the unpooled train data to the DataFrame dictionary. 
            if len(data) > 0:
                df['epoch'] += [i for i in range(self.epochs)]
                df['metric'] += [metric] * self.epochs
                # If it is a train metric, pool the values. 
                if 'train' in metric:
                    df['value'] += [np.mean(data[i:i + self.batches_per_epoch]) for i in range(0, len(data), self.batches_per_epoch)]
                else:
                    # Because I calculate validation loss at the beginning of each epoch, plus after the last epoch, I have one extra value.
                    # Remove the end loss calculation to account for this. 
                    df['value'] += data[:-1]

            check_df(df)

            if verbose: print(f'reporter.Reporter._get_info: Successfully added {metric} information to DataFrame.')

        return pd.DataFrame(df)

    def _get_info(self, metrics, verbose=False):
        '''Use the information returned by the train function to construct a DataFrame for plotting loss.'''

        # Make sure the reporter object is no longer actively being logged to. 
        assert not self.active

        df = {'batch':[], 'value':[], 'metric':[]}

        for metric, data in metrics.items():
            # First add the unpooled train data to the DataFrame dirctionary. 
            if len(data) > 0:
                df['value'] += data
                df['metric'] += [metric] * len(data)
                # Only train metrics are computed for each batch. 
                if 'train' in metric:
                    df['batch'] += [i for i in range(self.batches)]
                else:
                    df['batch'] += self.get_epoch_batches()

            check_df(df)

            if verbose: print(f'reporter.Reporter._get_info: Successfully added {metric} information to DataFrame.')

        return pd.DataFrame(df)

    def get_loss_info(self, verbose:bool=False, pool:bool=True) -> pd.DataFrame:
        '''Return a DataFrame containing loss information for plotting a training curve.'''
        if pool:
            return self._get_info_pooled(metrics=self.loss_metrics, verbose=verbose)
        else:
            return self._get_info(metrics=self.loss_metrics, verbose=verbose)

    def get_acc_info(self, verbose=False):
        return self._get_info(metrics=self.acc_metrics, verbose=verbose)

    # TODO: Should probably make similar accessors for each metric. 
    def get_test_losses(self):
        '''Return the test loss list from the reporter object.'''
        assert 'test_losses' in self.loss_metrics, 'No test_loss has been recorded.'
        assert len(self.loss_metrics['test_losses']) > 0, 'No test_loss has been recorded.'

        return self.loss_metrics['test_losses']

    # TODO: Should probably make similar accessors for each metric. 
    def get_test_accs(self):
        '''Return the test accuracy list from the reporter object.'''
        assert 'test_accs' in self.acc_metrics, 'reporter.Reporter.get_test_accs: No test_acc has been recorded.'
        assert len(self.acc_metrics['test_accs']) > 0, 'reporter.Reporter.get_test_accs: No test_loss has been recorded.'

        return self.acc_metrics['test_accs']
 
    def get_false_positive_rate(self):
        '''Calculate the false positive rate.'''
        tn, fp, fn, tp = self.get_confusion_matrix()
        fpr = fp / (fp + tn)

        return fpr
 
    def get_true_positive_rate(self):
        '''Calculate the true positive rate.'''
        tn, fp, fn, tp = self.get_confusion_matrix()
        tpr = tp / (tp + fn)
        
        return tpr
  


