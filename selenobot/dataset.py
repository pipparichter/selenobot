'''This file contains definitions for a Dataset object, and other associated functions.''' 
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from typing import List, Dict, NoReturn, Iterator
import subprocess
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which  Dataset objects which provide extra functionality for producing sequence embeddings
    and accessing information about selenoprotein content.'''
    
    def __init__(self, df:pd.DataFrame, embedder=None):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param embedder: The embedder to apply to the amino acid sequences in the DataFrame. 
            If None, it is assumed that the embeddings are already present in the file. 
        '''

        # Check to make sure all expected fields are present in the input DataFrame. 
        if embedder is not None:
            assert 'seq' in df.columns, f'dataset.Dataset.__init__: Input DataFrame missing required field seq.'
        assert 'label' in df.columns, f'dataset.Dataset.__init__: Input DataFrame missing required field label.'
        assert 'id' in df.columns, f'dataset.Dataset.__init__: Input DataFrame missing required field id.'

        if embedder is not None:
            self.embeddings = embedder(list(df['seq'].values))
            self.type = embedder.type # Type of data contained by the Dataset.
        else: # This means that the embeddings are already in the DataFrame (or at least, they should be)
            self.type = 'plm'
            self.embeddings = torch.from_numpy(df.drop(columns=['label', 'cluster', 'seq', 'id']).values).to(torch.float32)

        # Make sure the type of the tensor is the same as model weights.
        self.labels = torch.from_numpy(df['label'].values).type(torch.float32)
        self.ids = df['id'].values
        self.latent_dim = self.embeddings.shape[-1]

        self.length = len(df)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx:int) -> Dict:
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        label = self.labels[idx]
        embedding = self.embeddings[idx]
        id_ = self.ids[idx]
        return {'label':label, 'embedding':embedding, 'id':id_, 'idx':idx}

    def get_selenoprotein_indices(self) -> List[int]:
        '''Obtains the indices of selenoproteins in the Dataset.'''
        return list(np.where([']' in i for i in self.ids])[0])


# A BatchSampler should have an __iter__ method which returns the indices of the next batch once called.
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    '''A Sampler object which ensures that each batch has a similar proportion of selenoproteins to non-selenoproteins.
    This class is used in conjunction with PyTorch DataLoaders.'''

    # TODO: Probably add some checks here, unexpected bahavior might occur if things are evenly divisible by batch size, for example.
    def __init__(self, data_source:Dataset, batch_size:int=None,): 
        '''Initialize a custom BatchSampler object.
        
        :param data_source: A Dataset object containing the data to sample into batches. 
        :param batch_size: The size of the batches. 
        '''
        r = 0.5 # The fraction of truncated selenoproteins to include in each batch. 

        sel_idxs = data_source.get_selenoprotein_indices() # Get the indices of all tagged selenoproteins in the Dataset. 
        non_sel_idxs = np.array(list(set(range(len(data_source))) - set(sel_idxs))) # Get the indices of all non-selenoproteins in the dataset. 
        # This is the assumption made while selecting num_batches.
        assert len(sel_idxs) < len(non_sel_idxs), f'dataset.BalancedBatchSampler.__init__: Expecting fewer selenoproteins in the dataset than non-selenoproteins.'

        num_sel, num_non_sel = len(sel_idxs), len(non_sel_idxs) # Grab initial numbers of these for printing info at the end.

        # Shuffle the indices. 
        random.shuffle(sel_idxs)
        random.shuffle(non_sel_idxs)
        
        num_sel_per_batch = int(batch_size * r) # Number of selenoproteins in each batch. 
        num_non_sel_per_batch = batch_size - num_sel_per_batch # Number of full-length proteins in each batch. 
        num_batches = len(non_sel_idxs) // (batch_size - num_sel_per_batch) # Number of batches needed to cover the non-selenoproteins.
        
        non_sel_idxs = non_sel_idxs[:num_batches * num_non_sel_per_batch] # Random shuffled first, so removing from the end should not be an issue. 
        sel_idxs = np.resize(sel_idxs, num_batches * num_sel_per_batch) # Resize the array to the number of selenoproteins required for balanced batches.
        # Possibly want to shuffle these again? So later batches don't have the same selenoproteins as earlier ones.
        np.random.shuffle(sel_idxs)

        # Numpy split expects num_batches to be evenly divisible, and will throw an error otherwise. 
        sel_batches = np.split(sel_idxs, num_batches)
        non_sel_batches = np.split(non_sel_idxs, num_batches)
        self.batches = np.concatenate([sel_batches, non_sel_batches], axis=1)

        # Final check to make sure the number of batches and batch sizes are correct. 
        assert self.batches.shape == (num_batches, batch_size), f'dataset.BalancedBatchSampler.__init__: Incorrect batch dimensions. Expected {(num_batches, batch_size)}, but dimensions are {self.batches.shape}.'
        
        data_source.num_resampled = len(sel_idxs) - num_sel  
        data_source.num_removed = num_non_sel - len(non_sel_idxs)
        print(f'dataset.BalancedBatchSampler.__init__: Resampled {data_source.num_resampled} selenoproteins and removed {data_source.num_removed} non-selenoproteins to generate {num_batches} batches of size {batch_size}.')

    def __iter__(self) -> Iterator:
        return iter(self.batches)

    # Not sure if this should be the number of batches, or the number of elements.
    def __len__(self) -> int:
        '''Returns the number of batches.'''
        return len(self.batches)


def get_dataloader(
        dataset:Dataset, 
        batch_size:int=1024,
        balance_batches:bool=True) -> torch.utils.data.DataLoader:
    '''Create a DataLoader from a CSV file containing sequence and/or PLM embedding data.
    
    :param dataset: The Dataset used to generate the DataLoader. 
    :param batch_size: The size of the batches which the training data will be split into. 
    :param balance_batches: Whether or not to ensure that each batch has equal proportion of full-length and truncated proteins. 
    :return: A pytorch DataLoader object. 
    '''
    if balance_batches:
        # Providing batch_sampler will override batch_size, shuffle, sampler, and drop_last altogether.
        batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size)
        return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)




