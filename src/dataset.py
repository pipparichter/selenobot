'''This file contains definitions for a Dataset object, and other associated functions.''' 
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import embedders
from typing import List, NoReturn

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

PLM_LATENT_DIM = 1024 # Dimension of the PLM latent space.

def get_dataloader(
        path:str, 
        batch_size:int=1024,
        embedder:str=None) -> DataLoader:
    '''Create a DataLoader from a CSV of embedding data.
    
    args:
        - path: The path to the CSV file where the data is stored. 
        - batch_size: Batch size of the DataLoader. If None, no batches are used. 
        - embedder: A string specifying the embedder to apply to the data. 
    '''
    f = 'dataset.get_dataloader'
    
    data = pd.read_csv(path, usecols=['seq', 'id', 'label'])
    if embedder == 'aac':
        embeddings = embedders.AacEmbedder()(list(data['seq'].values))
    elif embedder == 'length':
        embeddings = embedders.LengthEmbedder()(list(data['seq'].values))
    elif embedder == 'plm': # For PLM embeddings, assume embeddings are in the file. 
        embeddings = pd.read_csv(path, usecols=[str(i) for i in range(PLM_LATENT_DIM)])
    else:
        raise ValueError(f'{f}: Embedder option must be one of aac, plm, or length.')

    # It makes more sense to pass in a path to the __.csv files, which have embedding information, as well as the sequences. 
    # Collect all the information into a single pandas DataFrame. 
    data = pd.concat([data, pd.DataFrame(embeddings)], axis=1)
    data = data.set_index('id')

    dataset = Dataset(data)

    # Providing batch_sampler will override batch_size, shuffle, sampler, and drop_last altogether. 
    # It is meant to define exactly the batch elements and their content.
    batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size, selenoprotein_fraction=0.25)
    return DataLoader(dataset, batch_sampler=batch_sampler)
    # return DataLoader(dataset, shuffle=True, batch_size=batch_size)


class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which provides easy access to sequence, label, and embedding data via the 
    overloaded __getitem__ method.'''
    
    def __init__(self, data:pd.DataFrame):
        '''Initializes an EmbeddingDataset from a pandas DataFrame containing embeddings and labels.'''
        assert 'seq' in data.columns, 'dataset.Dataset.__init__: Input DataFrame missing required field seq.' 
        assert 'label' in data.columns, 'dataset.Dataset.__init__: Input DataFrame missing required field label.' 

        # Make sure the type of the tensor is the same as model weights.
        self.embeddings_ = torch.from_numpy(data.drop(columns=['label', 'seq']).values).type(torch.float32)
        self.labels_ = torch.from_numpy(data['label'].values).type(torch.float32)
        self.ids_ = data.index
        self.latent_dim = self.embeddings_.shape[-1]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        return {'label':self.labels_[idx], 'emb':self.embeddings_[idx], 'id':self.ids_[idx], 'idx':idx}

    def get_selenoprotein_indices(self) -> List[int]:
        '''Obtains the indices of selenoproteins in the Dataset.'''
        return list(np.where([']' in id for id in self.ids_])[0])


# A BatchSampler should have an __iter__ method which returns the indices of the next batch once called.
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    '''A Sampler object which ensures that each batch has a similar proportion of selenoproteins to non-selenoproteins.'''

    # TODO: Probably add some checks here, unexpected bahavior might occur if things are evenly divisible by batch size, for example.
    def __init__(self, data_source:Dataset, batch_size:int=None, selenoprotein_fraction:float=0.25): # , shuffle=True):
        '''Initialize a custom BatchSampler object.'''
        f = 'dataset.BalancedBatchSampler.__init__'
        
        # super(BalancedBatchSampler, self).__init__()
        sel_idxs = data_source.get_selenoprotein_indices()
        non_sel_idxs = np.array(list(set(range(len(data_source))) - set(sel_idxs)))
        num_sel, num_non_sel = len(sel_idxs), len(non_sel_idxs) # Gran initial numbers of these. 

        random.shuffle(sel_idxs)
        random.shuffle(non_sel_idxs)
        
        num_sel_per_batch = int(batch_size * selenoprotein_fraction)
        num_non_sel_per_batch = batch_size - num_sel_per_batch

        # Number of batches needed to cover the non-selenoproteins. 
        num_batches = len(non_sel_idxs) // (batch_size - num_sel_per_batch)
        
        non_sel_idxs = non_sel_idxs[:num_batches * num_non_sel_per_batch]
        sel_idxs = np.resize(sel_idxs, num_batches * num_sel_per_batch)
        # Possibly want to shuffle these again? So later batches don't have the same selenoproteins as earlier ones.
        np.random.shuffle(sel_idxs)

        # Numpy split expects num_batches to be evenly divisible, and will throw an error otherwise. 
        sel_batches = np.split(sel_idxs, num_batches)
        non_sel_batches = np.split(non_sel_idxs, num_batches)
        
        self.batches = np.concatenate([sel_batches, non_sel_batches], axis=1)
        assert self.batches.shape == (num_batches, batch_size), f'{f}: Incorrect batch dimensions. Expected {(num_batches, batch_size)}, but dimensions are {self.batches.shape}.'

        print(f'{f}: Resampled {len(sel_idxs) - num_sel} selenoproteins and removed {num_non_sel - len(non_sel_idxs)} to generate {num_batches} batches of size {batch_size}.')

    def __iter__(self):
        return iter(self.batches)

    # Not sure if this should be the number of batches, or the number of elements.
    def __len__(self):
        return len(self.batches)
