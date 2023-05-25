'''
Definition of a Hidden Markov Model for classifying amino acid sequences (or the ESM embeddings of those sequences)
into two categories: incorrectly-truncated selenoproteins or regular proteins of comparable length. 
'''
# TODO: May eventually want to extend this model, or write a different model, which is capable of more than binary classification. 
# Perhaps, for example, we will want to detect artificially truncated proteins which are not selenoproteins/

import torch
import numpy as np

class HMM(torch.nn.Module):
    '''
    Binary Hidden Markov Model for predicting whether or not proteins are regular proteins, or incorrectly-truncated selenoproteins.
    '''
    