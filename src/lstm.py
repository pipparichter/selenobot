'''
Definition of an LSTM for classifying amino acid sequences (or the ESM embeddings of those sequences)
into two categories: incorrectly-truncated selenoproteins or regular proteins of comparable length. 
'''
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html 
import torch
import numpy as np
import tensorflow as tf

N_CLASSES = 2 # Only two classes for now -- truncated and not truncated. 

# NOTE: What will the vocab_size be for the ESM-generated embeddings? I think it might be like 320ish, but not entirely sure. 
# It still gives all float values, if I am not mistaken. What is a good way to convert this to a vocabulary?

# The ESM classifier adds a sequence classification/regression head (a linear layer on top of the pooled output).

class LSTM(nn.torch.Module):
    '''
    Class for applying a multi-layer long short-term memory classifier to sequence data. 
    '''

    # NOTE: Because sequence length is needed as an input when instantiating the model, might need to use padding. Make
    # sure all input sequences are the same length.     

    # NOTE: Do we want a bi-directional LSTM for this?
    def __init__(self, input_size, vocab_size, 
                batch_first=False, 
                hidden_size=2, 
                num_layers=1):

        '''
        All keyword arguments are passed into the torch.nn.LSTM constructor. 

        kwargs: 
            : batch_first (bool): False by default
            : input_size (int): The number of expected features (alphabet size). 22 by default, for the number of amino acids. 
            : hidden_size (int): The number of features in the hidden state. 
            : num_layers (int): 1 by default.  

        '''
        super(LSTM, self).__init__()

        # NOTE: How should we be picking the number of hidden dimensions?
        self.hidden_size = 32 
        
        self.lstm = torch.nn.LSTM(input_size, self.hidden_size)
        self.classifier = torch.nn.Linear(self.hidden_size, s)

        
        
        # Input to LSTM class is a tensor of shape (L, N, H_in) where L is sequence length, H_in is input size, and N is batch size (will be 1).
        # Also specify an h_0, tensor of shape (D*num_layers, N, H_out) for initial hidden state for each element in the input sequence. 
        # c_0 has dimensions (D*num_layers, N, H_cell) and specifies initial cell state for each element in the input sequence. Defaults to zeros. 

        self.model = torch.nn.LSTM()
        pass

        def predict():
            pass
        # It seems as though every element in the sequence (i.e. each amino acid position) gets its own embedding. 


