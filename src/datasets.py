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
from torch.utils.data import Dataset, DataLoader 
from utils.data import pd_from_fasta

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dataloader_from_csv(path, batch_size=64, type_='emb', **kwargs):
    '''Create a DataLoader from a CSV of embedding data.
    
    args:
        - path (str): The path to the CSV file where the data is stored. 
        - batch_size (int or None): Batch size of the DataLoader. If None, no batches are used. 
        - type_ (str): One of 'emb', 'seq'. Specifies which type of Dataset to use. 
        - **kwargs: keyword arguments passed into the Dataset.from_csv function, along with the path. 
    '''
    if type_ == 'emb':
        dataset = EmbeddingDataset.from_csv(path, **kwargs)
    elif type_ == 'seq':
        # Assume dataset is being loaded from a FASTA file. 
        dataset = SequenceDataset.from_fasta(path, **kwargs)
     
    # If the batch size is None, load the entire Dataset at once. 
    if batch_size is None:
        batch_size = len(dataset)

    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


class EmbeddingDataset(Dataset):
    
    def __init__(self, data):
        '''Initializes an EmbeddingDataset from a pandas DataFrame containing
        embeddings and labels.'''

        # Makes sense to store the embeddings as a separate array. 
        # Make sure the type of the tensor is the same as model weights. 
        self.embeddings_ = torch.from_numpy(data.drop(columns=['label']).values).type(torch.float32)
        self.labels_ = torch.from_numpy(data['label'].values).type(torch.float32)
        self.ids_ = np.ravel((data.index))

        self.latent_dim = self.embeddings_.shape[-1]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Returns an item from the EmbeddingDataset.'''
        return {'label':self.labels_[idx], 'data':self.embeddings_[idx], 'id':self.ids_[idx]}

    def from_csv(path):
        '''Load an EmbeddingDataset object from a CSV file.'''
        # Be picky about what is being read into the object. Eventually, we can 
        # add more checks to control what is loaded in (like type restrictions).
        data = pd.read_csv(path, index_col='id')
        return EmbeddingDataset(data)


class SequenceDataset(Dataset):

    def __init__(self, data):
        '''Instantiates a SequenceDataset using a pandas DataFrame which contains sequence and
        label information.'''

        # Make sure the type of the tensor is the same as model weights. 
        # self.sequences_ = torch.from_numpy(data['seq'].values).type(torch.float32)
        self.sequences_ = data['seq'].values
        self.labels_ = torch.from_numpy(data['label'].values)
        self.ids_ = np.array(list(data.index))

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Returns an item from the EmbeddingDataset.'''
        return {'label':self.labels_[idx], 'data':self.sequences_[idx]} # , 'id':self.ids_[idx]}
    
    def from_fasta(path, ids=None, labels=None):
        '''Load an EmbeddingDataset object from a FASTA file.'''

        assert len(ids) == len(labels)
        
        # Load in all the sequences to a DataFrame. 
        data = pd_from_fasta(path)

        if ids is not None:
            # Use the specified IDs to filter the data. 
            data = data[np.isin(data.index, ids)]

        # Add labels to the dataset. 
        if labels is not None:
            data = data.reindex(list(ids))
            data['label'] = labels

        return SequenceDataset(data) 

    # def from_csv(path, ids=None):
    #     '''Load an EmbeddingDataset object from a CSV file.'''

    #     # Load in all the sequences to a DataFrame. 
    #     data = pd.read_csv(path, index_col='id')

    #     if ids is not None:
    #         # Use the specified IDs to filter the data. 
    #         data = data[np.isin(data.index, ids)]

    #     return SequenceDataset(data)