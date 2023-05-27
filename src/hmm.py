'''
Definition of a Hidden Markov Model for classifying amino acid sequences (or the ESM embeddings of those sequences)
into two categories: incorrectly-truncated selenoproteins or regular proteins of comparable length. 

Need to talk to Josh, but it seems like an LSTM model might be a better choice. 
https://www.biorxiv.org/content/10.1101/626507v4.full (will be valuable to read through)
https://towardsdatascience.com/hidden-markov-model-applied-to-biological-sequence-373d2b5a24c 
'''

# NOTE: I am kind of skeptical about the memoryless assumption inherent to the HMM... This seems like it wouldn't be good for
# categorizing sequences, as the entire thing matters?

# TODO: May eventually want to extend this model, or write a different model, which is capable of more than binary classification. 
# Perhaps, for example, we will want to detect artificially truncated proteins which are not selenoproteins. 

# Based on code from https://colab.research.google.com/drive/1IUe9lfoIiQsL49atSOgxnCmMR_zJazKI#scrollTo=aZbW6Pj0og7K 

import torch
import numpy as np


# This model will have two observation categories: truncated and full-length. These are the states which the emission probabilities
# will correspond to. 
class HMM(torch.nn.Module):
    '''
    Binary Hidden Markov Model for predicting whether or not proteins are regular proteins, or incorrectly-truncated selenoproteins.
    '''
    def __init__(self):
        super(HMM, self).__init__()

        self.n = 2 # Number of observations
        # NOTE: Do I also include something for a stop signal here?
        self.m = 22 # Number of states... I think this is the number of amino acids?



class HMMTransitionModel(torch.nn.Module):
    '''
    '''
    def __init__(self, m):
        super(HMMTransitionModel, self).__init__()

        self.m = m # Number of possible states. It does seem as though this is the length of the alphabet. 
        # Type of tensor which, when assigned as Module attributes, are automatically added to the list of its parameters
        self.matrix = torch.nn.Parameter(torch.randn(m, m)) # Transition matrix.

class HMMEmissionModel(torch.nn.Module):
    '''
    '''
    def __init__(self, m, n):
        super(HMMEmissionModel, self).__init__()

        self.m = m
        self.n = n
        # Emission matrix maps the hidden states to observations (i.e. predicts the observation)
        self.matrix = torch.nn.Parameter(torch.randn(m, n)) # Emission matrix.


