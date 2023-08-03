'''
One of the questions which arose was, understandably, why dooes the linear classifier preform so well?
The suspicion is that it is simply picking up on amino acid composition, and that this is enough to distinguish
between truncated selenoproteins and short proteins, in this particular case. This "model" (which is not really a model)
is designed to test this theory.
'''
from torch.nn.functional import cross_entropy, binary_cross_entropy
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

# TODO: Possibly re-name the tokenizer to like, AAC tokenizer. 

# Tokenizer class to mimic the HuggingFace interface.   
# This class is used within SequenceDataset class. This class takes the entire DataFrame upon instantiation,
# and handles the labels this way. 
class BenchmarkTokenizer():
    def __init__(self): #, padding=False):
        '''

        '''

        aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
        self.map_ = {aas[i]: i + 1 for i in range(len(aas))} # Amino acid to integer map. 
        self.map_['<eos>'] = 0 # Add a termination token. 
        # self.padding = padding # Whether or not to zero-pad the data. 
        

    def __call__(self, data): # How to make the tokenizer callable. 
        '''
        args:
            - data (list): A list of amino acid sequences. 
        '''
        # Map each amino acid to an integer using self.map_
        mapped = [np.array([self.map_[res] for res in seq]) for seq in data]

        encoded_seqs = []
        # Create one-hot encoded array and sum along the rows (for each mapped sequence). 
        for mapped_seq in mapped:

            seq_length = len(mapped_seq)

            # Create an array of zeroes with the number of map elements as columns. 
            encoded_seq = np.zeros(shape=(seq_length, len(self.map_)))
            encoded_seq[np.arange(seq_length), mapped_seq] = 1
            encoded_seq = np.sum(encoded_seq, axis=0)
            # Now need to normalize according to sequence length. 
            encoded_seq = encoded_seq / seq_length

            encoded_seqs.append(encoded_seq.tolist())
       
        # encoded_seqs is now a 2D list. Convert to a tensor for the sake of compatibility
        return {'input_ids':torch.tensor(encoded_seqs)}


class BenchmarkClassifier():
    '''
    '''
    def __init__(self):

        # self.n_components = n_components # For the PCA model
        self.logreg = LogisticRegression() # (positive=True)

    def __call__(self, **kwargs):
        '''
        Just calls the predict function. 
        '''
        return self.predict(**kwargs)

    def fit(self, data, labels):
        '''
        Fits the underlying logistic regression model to input data. In this case, labels must be specified. 

        args:
            - data (np.array)
            - labels (np.array)
        '''
        self.logreg.fit(data, labels) # Should modify the object. 

    def predict(self, data, labels=None):
        '''
        Uses the fitted logistic regression model to predict the labels of the input data. 
        If labels are given, the test accuracy and lost are returned. 

        args:
            - data (np.array)
            - labels (np.array)
        '''
        # Pass the PCA outputs into the already-fitted logistic regression thing. 
        preds = self.logreg.predict(data)
        
        # Calculate loss if labels are specified. 
        loss = None
        # NOTE: For loss, maybe I should be outputting specific probabilities, not labels. 
        if labels is not None:
            probs = torch.DoubleTensor(self.logreg.predict_proba(labels)[:, 1])
            # Might need to convert things to tensors for this to work.  
            loss = binary_cross_entropy(probs, torch.DoubleTensor(labels)).item()
        
        return preds, loss


