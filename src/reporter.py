'''Defining a class used to more easily manage data reporting on the performance of a Classifier.'''

import numpy as np
import pandas as pd
import pickle
import sklearn



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
        self.confusion_matrices = []

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

    def add_train_loss(self, loss):
        '''Add a train_loss to the internal list.'''
        self.check_active()
        check_type(loss.item(), float)
        if 'train_losses' not in self.loss_metrics:
            self.loss_metrics['train_losses'] = []
        self.loss_metrics['train_losses'].append(loss.item())

    def add_train_acc(self, acc):
        '''Add a train_acc to the internal list.'''
        self.check_active()
        check_type(acc, float)
        if 'train_accs' not in self.acc_metrics:
            self.acc_metrics['train_accs'] = []
        self.acc_metrics['train_accs'].append(acc)

    def add_train_metrics(self, loss, acc):
        self.add_train_loss(loss)
        self.add_train_acc(acc)

    def add_val_loss(self, loss):
        '''Add a val_loss to the internal list.'''
        self.check_active()
        check_type(loss.item(), float)
        if 'val_losses' not in self.loss_metrics:
            self.loss_metrics['val_losses'] = []
        self.loss_metrics['val_losses'].append(loss.item())

    def add_val_acc(self, acc):
        '''Add a val_acc to the internal list.'''
        self.check_active()
        check_type(acc, float)
        if 'val_accs' not in self.acc_metrics:
            self.acc_metrics['val_accs'] = []
        self.acc_metrics['val_accs'].append(acc)

    def add_val_metrics(self, loss, acc):
        self.add_val_loss(loss)
        self.add_val_acc(acc)

    def add_test_loss(self, loss):
        '''Add a train_loss to the internal list.'''
        self.check_active()
        check_type(loss.item(), float)
        if 'test_losses' not in self.loss_metrics:
            self.loss_metrics['test_losses'] = []
        self.loss_metrics['test_losses'].append(loss.item())

    def add_test_acc(self, acc):
        '''Add a train_acc to the internal list.'''
        self.check_active()
        check_type(acc, float)
        if 'test_accs' not in self.acc_metrics:
            self.acc_metrics['test_accs'] = []
        self.acc_metrics['test_accs'].append(acc)

    def add_test_metrics(self, loss, acc):
        self.add_test_loss(loss)
        self.add_test_acc(acc)

    def get_epoch_batches(self):
        '''Get the batch numbers where each new epoch begins.'''
        assert self.batches_per_epoch is not None, 'batches_per_epoch attribute has not been set.'
        step = self.batches_per_epoch
        return [i for i in range(step, self.batches+step, step)]
    
    def add_confusion_matrix(self, tn:int, fp:int, fn:int, tp:int) -> None:
        '''Add new confusion matrix data to the internal list.'''
        self.check_active()
        self.confusion_matrices.append((tn, fp, fn, tp))

    def get_confusion_matrix(self) -> tuple:
        '''Return the first confusion matrix stored in the internal list.'''
        if len(self.confusion_matrices) > 1:
            print(f'There are {len(self.confusion_matrices)} stored in the Reporter object. Returning the first entry.')
        assert len(self.confusion_matrices) > 0, 'No confusion matrices stored in the Reporter.'
        return self.confusion_matrices[0]

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

            if verbose: print(f'reporter.Reporter._geSuccessfully added {metric} information to DataFrame.')

        return pd.DataFrame(df)

    def get_loss_info(self, verbose=False):
        return self._get_info(metrics=self.loss_metrics, verbose=verbose)

    def get_acc_info(self, verbose=False):
        return self._get_info(metrics=self.acc_metrics, verbose=verbose)

    # TODO: Should probably make similar accessors for each metric. 
    def get_test_losses(self):
        '''Return the test loss list from the reporter object.'''
        assert 'test_losses' in self.loss_metrics, 'No test_loss has been recorded.'
        assert len(self.loss_metrics['test_losses']) > 0, 'No test_loss has been recorded.'

        return self.loss_metrics['test_losses']
        



