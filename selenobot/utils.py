'''Utility functions for reading and writing FASTA and CSV files, amongst other things.'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from typing import Dict, NoReturn, List
import configparser
import pickle
import subprocess
import json
from collections import OrderedDict
import torch

# Define some important directories...
cwd, _ = os.path.split(os.path.abspath(__file__))
ROOT_DIR = os.path.join(cwd, '..')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results') # Get the path where results are stored.
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Get the path where results are stored. 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts') # Get the path where results are stored.


def to_numeric(n:str):
    '''Try to convert a string to a numerical data type. Used when 
    reading in header strings from files.'''
    try: 
        n = int(n)
    except:
        try: 
            n = float(n)
        except:
            pass
    return n


class NumpyEncoder(json.JSONEncoder):
    '''Encoder for converting numpy data types into types which are JSON-serializable. Based
    on the tutorial here: https://medium.com/@ayush-thakur02/understanding-custom-encoders-and-decoders-in-pythons-json-module-1490d3d23cf7'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # For saving state dicts, which are dictionaries of tensors. 
        if isinstance(obj, OrderedDict):
            new_obj = OrderedDict() # Make a new ordered dictionary. 
            for k, v in obj.items():
                new_obj[k] = v.tolist()
            return new_obj 
        if isinstance(obj, torch.Tensor):
            return obj.tolist()

        return super(NumpyEncoder, self).default(obj)


# class TrainInfo():
#     '''A class for managing the results of training a Classifier. The information stored in this
#     object can be used to plot training curves.'''

#     def __init__(self, epochs:int=None, lr:float=None, batches_per_epoch:int=None):
#         '''Initialize a TrainInfo object.
        
#         :param epochs: The number of epochs for which the model is trained. 
#         :param lr: The learning rate at which the model was trained.
#         :batches_per_epoch: The number of batches in each epoch during training. 
#         '''
#         super().__init__()

#         self.epochs = epochs
#         self.best_epoch = None
#         self.lr = lr
#         self.batches_per_epoch = batches_per_epoch

#         # Only store the losses and accuracies, as storing the entire sets of outputs and
#         # targets for each batch seems like too much. 
#         self.train_losses = []
#         self.val_losses = []


#     def get_training_curve_data(self) -> pd.DataFrame:
#         '''Organizes the data contained in the info into a pandas DataFrame to
#         make it convenient to plot training curves.'''
#         assert len(self.train_losses) == len(self.val_losses), 'TrainInfo.get_training_curve_data: The number of recorded validation and training losses should be equal.'
#         n = len(self.train_losses) # Total number of recorded values. 

#         data = {}
#         data['metric'] = ['training loss'] * n + ['validation loss'] * n
#         data['epoch'] = list(range(n)) * 2
#         data['value'] = self.train_losses + self.val_losses

#         return pd.DataFrame(data)


# class TestInfo():
#     '''A class for managing the results of evaluating a Classifier on test data.'''

#     def __init__(self, outputs, targets):
#         '''Initialize a TestInfo object, which stores data for assessing model 
#         performance on a Dataset object.
        
#         :param threshold: The threshold to apply to the model outputs when computing predictions.
#         '''
#         # Make sure outputs and targets are stored as one-dimensional numpy arrays. 
#         self.outputs = outputs.detach().numpy().ravel()

#         if targets is not None:
#             self.targets = targets.detach().numpy().ravel()
#         else:
#             self.targets = None

#         self.loss = None


#     def get_balanced_accuracy(self, threshold:float=0.5) -> float:
#         '''Applies a threshold to the model outputs, and uses it to compute a balanced accuracy score.'''
#         outputs = self.apply_threshold(threshold=threshold)
#         # Compute balanced accuracy using a builtin sklearn function. 
#         return sklearn.metrics.balanced_accuracy_score(self.targets, outputs)

#     def get_auc(self) -> float:
#         '''Computes the AUC score for the contained outputs and targets.'''
#         return sklearn.metrics.roc_auc_score(self.targets, self.outputs)
    
#     def get_expected_calibration_error(self, nbins:int=10) -> float:
#         '''Calculate the expected calibration error of the test predictions.'''
#         # First, sort the predictions into bins. 
#         bins = np.linspace(0, 1, num=nbins + 1, endpoint=True)
#         # Setting right=False means the lower bound is inclusive, and the upper bound is non-inclusive. 
#         bin_idxs = np.digitize(self.outputs, bins, right=False)

#         err = 0
#         n = len(self.outputs) # Get the total number of outputs. 
#         # Should note that bin indices seem to start at 1. 0 is for things that are outside of the bin range. 
#         for b in range(1, len(bins)):
#             m = sum(bin_idxs == b) # Get the number of things in the bin.

#             if m > 0:
#                 bin_acc = self.targets[bin_idxs == b].mean()
#                 bin_conf = self.outputs[bin_idxs == b].mean()
#                 err += (m/n) * abs(bin_acc - bin_conf)
        
#         return err


    









