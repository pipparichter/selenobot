'''
This file contains definitions for different Datasets, all of which inherit from the 
torch.utils.data.Dataset class (and are therefore compatible with a Dataloader)
'''

import pandas as pd
import numpy as np
import torch
import h5py
import os
import torch
from torch.utils.data import DataLoader
from utils.data import pd_from_fasta

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader(path, batch_size=1024):
    '''Create a DataLoader from a CSV of embedding data.
    
    args:
        - path (str): The path to the CSV file where the data is stored. 
        - batch_size (int or None): Batch size of the DataLoader. If None, no batches are used. 
    '''
    dataset = Dataset.from_csv(path)
     
    # If the batch size is None, load the entire Dataset at once. 
    if batch_size is None:
        batch_size = len(dataset)

    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        '''Initializes an EmbeddingDataset from a pandas DataFrame containing
        embeddings and labels.'''

        # Makes sense to store the embeddings as a separate array. 
        # Make sure the type of the tensor is the same as model weights. 
        self.embeddings_ = torch.from_numpy(data.drop(columns=['label', 'seq']).values).type(torch.float32)
        self.labels_ = torch.from_numpy(data['label'].values).type(torch.float32)
        self.ids_ = data.index
        self.sequences_ = np.array(data['seq'])

        self.latent_dim = self.embeddings_.shape[-1]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Returns an item from the EmbeddingDataset.'''
        return {'label':self.labels_[idx], 'emb':self.embeddings_[idx], 'id':self.ids_[idx], 'seq':self.sequences_[idx]}

    def from_csv(path):
        '''Load an EmbeddingDataset object from a CSV file.'''
        data = pd.read_csv(path, index_col='id')
        return Dataset(data)



    # def from_csv(path, ids=None):
    #     '''Load an EmbeddingDataset object from a CSV file.'''

    #     # Load in all the sequences to a DataFrame. 
    #     data = pd.read_csv(path, index_col='id')

    #     if ids is not None:
    #         # Use the specified IDs to filter the data. 
    #         data = data[np.isin(data.index, ids)]

    #     return SequenceDataset(data)