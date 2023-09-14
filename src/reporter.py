'''Defining a class used to more easily manage data reporting on the performance of a Classifier.'''

import numpy as np
import pandas as pd
import pickle


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

        self.loss_metrics = ['val_losses', 'train_losses', 'test_losses']
        self.acc_metrics = ['val_acc', 'train_acc', 'test_acc']

        self.active = False
        
        self.epochs = epochs
        self.lr = lr
        self.bce_loss_weight = bce_loss_weight

        self.val_losses, self.val_accs = [], []
        self.train_losses, self.train_accs = [], []

        # TODO: Add stuff for test losses. 

    def check_active(self):
        '''Check if the Reporter is active.'''
        assert self.active, 'Reporter object must be active to add metrics.'


    def open(self):
        self.active = True

    def close(self):
        self.active = False
        # Now that logging is complete, load the number of batches which have been processed, if any. 
        self.batches = len(self.train_losses)
    
    def set_batches_per_epoch(self, batches_per_epoch):
        check_type(batches_per_epoch, int)
        self.batches_per_epoch = batches_per_epoch

    def add_train_loss(self, loss):
        '''Add a train_loss to the internal list.'''
        self.check_active()
        # The input loss is expected to be a tensor, so make sure to convert to a float. 
        self.train_losses.append(loss.item())

    def add_train_acc(self, acc):
        '''Add a train_acc to the internal list.'''
        self.check_active()
        check_type(acc, float)
        self.train_accs.append(acc)

    def add_train_metrics(self, loss, acc):
        self.add_train_loss(loss)
        self.add_train_acc(acc)

    def add_val_loss(self, loss):
        '''Add a val_loss to the internal list.'''
        self.check_active()
        # The input loss is expected to be a tensor, so make sure to convert to a float. 
        self.val_losses.append(loss.item())

    def add_val_acc(self, acc):
        '''Add a val_acc to the internal list.'''
        self.check_active()
        check_type(acc, float)
        self.val_accs.append(acc)

    def add_val_metrics(self, loss, acc):
        self.add_val_loss(loss)
        self.add_val_acc(acc)

    # def add_test_loss(self, test_loss):
    #     '''Add a test_loss to the internal list.'''
    #     self.check_active()
    #     # The input loss is expected to be a tensor, so make sure to convert to a float. 
    #     self.test_loss.append(test_loss.item())

    # def add_test_acc(self, test_acc):
    #     '''Add a test_acc to the internal list.'''
    #     self.check_active()build_loss_df(info, pool=pool)
    #     # The input loss is probably a tensor, so make sure to convert to a float. 
    #     assert type(test_acc) == float
    #     self.test_acc.append(test_acc)

    def get_epoch_batches(self):
        '''Get the batch numbers where each new epoch begins.'''
        step = self.batches_per_epoch
        return [i for i in range(step, self.batches+step, step)]

    def _get_info(self, metrics, verbose=False):
        '''Use the information returned by the train function to construct a DataFrame for plotting loss.'''

        # Make sure the reporter object is no longer actively being logged to. 
        assert not self.active

        df = {'batch':[], 'loss':[], 'metric':[]}

        for metric in self.loss_metrics:
            # First add the unpooled train data to the DataFrame dirctionary. 
            if len(self.__getattr__(metric)) > 0:
                df['loss'] += self.__getattr__(metric)
                df['metric'] += [metric] * len(self.__getattr__(metric))

                # Only train metrics are computed for each batch. 
                if 'train' in metric:
                    df['batch'] += [i for i in range(self.batches)]
                else:
                    df['batch'] += self.get_epoch_batches()

            check_df(df)

            if verbose: print(f'Successfully added {metric} information to DataFrame.')

        return pd.DataFrame(df)

    def get_loss_info(self, verbose=False):
        return self._get_info(metrics=self.loss_metrics, verbose=verbose)

    def get_loss_info(self, verbose=False):
        return self._get_info(metrics=self.acc_metrics, verbose=verbose)


