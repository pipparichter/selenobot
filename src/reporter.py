'''Defining a class used to more easily manage data reporting on the performance of a Classifier.'''

import tqdm
import numpy as np
import pandas as pd


def check_df(df):
    '''Quick little function for checking DataFrames under construction.'''
    msg = 'plot.check_df: Column lengths are mismatched, ' + ' '.join(f'len({key})={len(val)}' for key, val in df.items())
    # assert (len(df['batch']) == len(df['loss'])) and (len(df['loss']) == len(df['label'])), msg
    
    length = len(list(df.values())[0])
    assert np.all([len(df[key]) == length for key in df.keys()]), msg

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
        self.pbar = tqdm.tqdm(range(self.epochs), desc='reporter.Reporter.open')

    def close(self):
        self.active = False
        # Now that logging is complete, load the number of batches which have been processed, if any. 
        self.batches = len(self.train_loss)
        self.pbar.close() # Close out the progress bar. 
    
    def set_batches_per_epoch(self, batches_per_epoch):
        assert type(batches_per_epoch) == int
        self.batches_per_epoch = batches_per_epoch

    def add_train_loss(self, train_loss):
        '''Add a train_loss to the internal list.'''
        self.check_active()
        # The input loss is expected to be a tensor, so make sure to convert to a float. 
        self.train_loss.append(train_loss.item())

    def add_train_acc(self, train_acc):
        '''Add a train_acc to the internal list.'''
        self.check_active()
        # The input loss is probably a tensor, so make sure to convert to a float. 
        assert type(train_acc) == float
        self.train_acc.append(train_acc)

    def add_train_metrics(self, train_loss, train_acc):
        self.add_train_loss(train_loss)
        self.add_train_acc(train_acc)

    def add_val_loss(self, val_loss):
        '''Add a val_loss to the internal list.'''
        self.check_active()
        # The input loss is expected to be a tensor, so make sure to convert to a float. 
        self.val_loss.append(val_loss.item())

    def add_val_acc(self, val_acc):
        '''Add a val_acc to the internal list.'''
        self.check_active()
        # The input loss is probably a tensor, so make sure to convert to a float. 
        assert type(val_acc) == float
        self.val_acc.append(val_acc)
        self.pbar.set_postfix({'val_acc':np.round(val_acc, 2)})

    def add_val_metrics(self, val_loss, val_acc):
        self.add_val_loss(val_loss)
        self.add_val_acc(val_acc)

    # def add_test_loss(self, test_loss):
    #     '''Add a test_loss to the internal list.'''
    #     self.check_active()
    #     # The input loss is expected to be a tensor, so make sure to convert to a float. 
    #     self.test_loss.append(test_loss.item())

    # def add_test_acc(self, test_acc):
    #     '''Add a test_acc to the internal list.'''
    #     self.check_active()
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


