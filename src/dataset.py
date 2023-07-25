'''
Definition of the SequenceDataset object, which is designed to interface between the files generated in the
/protex/data/data.py file and PyTorch models. Later added functionality for compatibility with non-ML
models, like that defined in /protex/src/bench.py. 
'''

import torch
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels=None, tokenizer=None, embeddings=None, **kwargs):
        '''
        Initialize the SequenceDataset oject. 

        args:
            - data (pd.DataFrame): The amino acid sequence data. Should have columns
                'seq' at minimum, unless the data is already tokenized. 
            - tokenizer (torch.Tokenizer): A tokenizer for converting the sequence data 
                to numerical data. 
            - embeddings (pd.DataFrame): Pre-generated embeddings, if applicable. 
            - kwargs: Specs to be passed into the Tokenizer before it is called on the data. 
        '''
        if tokenizer is not None:
            # Tokenizer should always return a dictionary. One of the keys
            # should be input_ids, with a tensor as the value.  
            self.data = tokenizer(list(data['seq']), **kwargs)
        else: # If no tokenizer is specified, probably passing in pre-embedded data. 
            self.data = {'input_ids':torch.Tensor(data.drop(columns=['index']).values)}

        # Store as tensor for compatibility Pytorch stuff. 
        self.labels = labels if labels is None else torch.Tensor(labels)

        if 'label' in data.columns:
            self.labels = torch.tensor(data['label'])

        self.length = len(data)

        self.embeddings = None
        # Load in pre-generated embedding data, if available. Indices should align with dataset indices. 
        if embeddings is not None:
            indices = embeddings['index'].values
            embeddings = embeddings.drop(columns=['index']).values
            # Sort the embeddings according to the indices. Should already be no duplicates.  
            embeddings = embeddings[np.argsort(indices)]
            self.embeddings = torch.tensor(embeddings)

    def __getitem__(self, idx):
        # Get an item from the dataset 

        item = {key: val[idx] for key, val in self.data.items()}

        if self.labels is not None:
            item['labels'] = self.labels[idx]
        if self.embeddings is not None:
            item['embeddings'] = self.embeddings[idx]

        item['index'] = idx # Include the index so you can match with the original dataset. 

        return item

    def __len__(self):
        return self.length

    def get_data(self):
        '''
        Return underlying data as a numpy array. 
        '''
        return self.data['input_ids'].numpy()

    def get_labels(self):
        '''
        Return labels as a numpy array, if labels are provided. 
        '''
        if self.labels is not None:
            return self.labels.numpy()
        else:
            return None

    def get_embeddings(self):
        '''
        Return embeddings as a numpy array, if labels are provided. 
        '''
        if self.embeddings is not None:
            return self.embeddings.numpy()
        else:
            return None