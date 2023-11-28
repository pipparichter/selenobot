'''This file contains definitions for a Dataset object, and other associated functions.''' 
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import embedders
from typing import List
import subprocess
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(
        path:str, 
        batch_size:int=16,
        embedder:str=None,
        balance_batches:bool=True,
        selenoprotein_fraction:float=0.5,
        verbose:bool=True) -> torch.utils.data.DataLoader:
    '''Create a DataLoader from a CSV of embedding data.
    
    args:
        - path: The path to the CSV file where the data is stored. 
        - batch_size: Batch size of the DataLoader.  
        - embedder: One of length, aac, or plm. Specifies the embedding method used.
        - balance_batches: Whether or not to use the BalancedBatchSampler.
        - selenoprotein_fraction: A number between 0 and 1. Specifies the proportion of selenoproteins in each batch. 
        - verbose: Whether or not to print out assorted runtime things. 

    '''
    f = 'dataset.get_dataloader'
    # Collect all the information into a single pandas DataFrame. Avoid loading all embeddings if not needed.
    data = pd.read_csv(path, usecols=None if embedder == 'plm' else ['seq', 'label', 'id'])
    dataset = Dataset(data, embedder=embedder)

    if balance_batches:
        # Providing batch_sampler will override batch_size, shuffle, sampler, and drop_last altogether.
        batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size, selenoprotein_fraction=selenoprotein_fraction, verbose=verbose)
        return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

# def get_low_memory_dataloader


class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which provides easy access to sequence, label, and embedding data via the 
    overloaded __getitem__ method.'''
    
    def __init__(self, data:pd.DataFrame, embedder:str=None):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        args:
            - data: A pandas DataFrame loaded from the data/detect subdirectory. 
            - embedder: One of length, aac, or plm. Specifies the embedding method used.
        '''
        f = 'dataset.Dataset.__init__'
        if embedder is not None:
            assert 'seq' in data.columns, f'{f}: Input DataFrame missing required field seq.'
        assert 'label' in data.columns, f'{f}: Input DataFrame missing required field label.'
        assert 'id' in data.columns, f'{f}: Input DataFrame missing required field id.'

        if embedder == 'aac':
            self.embeddings = embedders.AacEmbedder()(list(data['seq'].values))
        elif embedder == 'length':
            self.embeddings = embedders.LengthEmbedder()(list(data['seq'].values))
        elif embedder == 'plm': # For PLM embeddings, assume embeddings are in the file. 
            self.embeddings = torch.from_numpy(data.drop(columns=['label', 'cluster', 'seq', 'id']).values).to(torch.float32)
        else:
            raise ValueError(f'{f}: Embedder option must be one of aac, plm, or length.')

        # Make sure the type of the tensor is the same as model weights.
        self.labels = torch.from_numpy(data['label'].values).type(torch.float32)
        self.ids = data['id']
        self.latent_dim = self.embeddings.shape[-1]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        return {'label':self.labels[idx], 'emb':self.embeddings[idx], 'id':self.ids[idx], 'idx':idx}

    def get_labels(self):
        '''Accessor for the labels associated with the Dataset.'''
        return self.labels

    def get_embeddings(self):
        '''Accessor function for the embeddings stored in the Dataset.'''
        return self.embeddings

    def get_selenoprotein_indices(self) -> List[int]:
        '''Obtains the indices of selenoproteins in the Dataset.'''
        return list(np.where([']' in id for id in self.ids])[0])


# A BatchSampler should have an __iter__ method which returns the indices of the next batch once called.
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    '''A Sampler object which ensures that each batch has a similar proportion of selenoproteins to non-selenoproteins.'''

    # TODO: Probably add some checks here, unexpected bahavior might occur if things are evenly divisible by batch size, for example.
    def __init__(self, 
        data_source:Dataset, 
        batch_size:int=None, 
        selenoprotein_fraction:float=0.5,
        verbose:bool=True): # , shuffle=True):
        '''Initialize a custom BatchSampler object.'''
        f = 'dataset.BalancedBatchSampler.__init__'
        
        # super(BalancedBatchSampler, self).__init__()
        
        sel_idxs = data_source.get_selenoprotein_indices()
        non_sel_idxs = np.array(list(set(range(len(data_source))) - set(sel_idxs)))
        # This is the assumption made while selecting num_batches, and removing from the non_sel_idxs list.
        assert len(sel_idxs) < len(non_sel_idxs), f'{f}: Expecting fewer selenoproteins in the dataset than non-selenoproteins.'

        num_sel, num_non_sel = len(sel_idxs), len(non_sel_idxs) # Grab initial numbers of these for printing info at the end.

        random.shuffle(sel_idxs)
        random.shuffle(non_sel_idxs)
        
        num_sel_per_batch = int(batch_size * selenoprotein_fraction)
        num_non_sel_per_batch = batch_size - num_sel_per_batch

        # Number of batches needed to cover the non-selenoproteins. 
        num_batches = len(non_sel_idxs) // (batch_size - num_sel_per_batch)
        
        non_sel_idxs = non_sel_idxs[:num_batches * num_non_sel_per_batch] # Random shuffled first, so removing from the end should not be an issue. 
        sel_idxs = np.resize(sel_idxs, num_batches * num_sel_per_batch)
        # Possibly want to shuffle these again? So later batches don't have the same selenoproteins as earlier ones.
        np.random.shuffle(sel_idxs)

        # Numpy split expects num_batches to be evenly divisible, and will throw an error otherwise. 
        sel_batches = np.split(sel_idxs, num_batches)
        non_sel_batches = np.split(non_sel_idxs, num_batches)
 
        self.batches = np.concatenate([sel_batches, non_sel_batches], axis=1)

        assert self.batches.shape == (num_batches, batch_size), f'{f}: Incorrect batch dimensions. Expected {(num_batches, batch_size)}, but dimensions are {self.batches.shape}.'
        
        # self.num_resampled = len(sel_idxs) - num_sel 
        # self.num_removed = num_non_sel - len(non_sel_idxs)
        data_source.num_resampled = len(sel_idxs) - num_sel  
        data_source.num_removed = num_non_sel - len(non_sel_idxs)
        if verbose:
            print(f'{f}: Resampled {data_source.num_resampled} selenoproteins and removed {data_source.num_removed} non-selenoproteins to generate {num_batches} batches of size {batch_size}.')

    def __iter__(self):
        return iter(self.batches)

    # Not sure if this should be the number of batches, or the number of elements.
    def __len__(self):
        return len(self.batches)



# class LowMemoryDataset(torch.utils.data.Dataset):
#     '''A map-style dataset which avoids loading the entire DataFrame stored in data_path at once. Works
#     well when using high-dimensional embeddings, like PLM embeddings.'''

#     def __init__(self, data_path:str):
#         '''Initializes a LowMemoryDataset from a data_path.'''
#         f = 'datasets.LowMemoryDataset.__init__'

#         self.data_path = data_path
#         # Executing a bash command is probably faster than readlines, considering how big the file is. 
#         line_count = int(subprocess.run(f'wc -l {data_path}', shell=True, check=True, capture_output=True, text=True).stdout.split()[0])
#         self.length = line_count - 1 # Subtract 1 from the file line count to account for the header row.

#         # Read in the columns using a bash command, which is probably faster than pandas.  
#         self.columns = subprocess.run(f'head -1 {data_path}', shell=True, check=True, capture_output=True, text=True).stdout.strip().split(',')
#         assert 'seq' in self.columns, f'{f}: Input DataFrame missing required field seq.'
#         assert 'label' in self.columns, f'{f}: Input DataFrame missing required field label.'
#         assert 'id' in self.columns, f'{f}: Input DataFrame missing required field id.'
#         self.latent_dim = len(self.columns) - 3

#     def __len__(self):
#         '''The length of the DataFrame at data_path(does not include the header in the CSV file).'''
#         return self.length

#     # def __getitem__(self, idx):
#     #     '''Loads a specific row from the pandas DataFrame stored at self.data_path.'''
#     #     f = 'datasets.LowMemoryDataset.__getitem__'

#     #     ti = time.perf_counter()
#     #     skiprows = [i + 1 for i in range(self.length) if (i + 1) != idx] # Make sure not to skip the header row.
#     #     data = pd.read_csv(self.data_path, skiprows=skiprows)
#     #     assert len(data) == 1, f'{f}: {len(data)} rows read from {self.data_path}. Expected 1.'
        
#     #     item = {col:data[col].item() for col in ['label', 'seq', 'id']}
#     #     item['emb'] = torch.from_numpy(data.drop(columns=['label', 'seq', 'id']).values).to(torch.float32) # Add the embedding.
#     #     tf = time.perf_counter()
#     #     print(f'{f}: Accessed row at index {idx} in {tf - ti}')
#     #     return item

#     # def __getitem__(self, idx):
#     #     '''Loads a specific row from the pandas DataFrame stored at self.data_path.'''
#     #     f = 'datasets.LowMemoryDataset.__getitem__'
#     #     ti = time.perf_counter()
#     #     data = subprocess.run(f"sed -ne '{idx + 1}p;{idx + 1}q' {self.data_path}", shell=True, check=True, capture_output=True, text=True).stdout.strip().split(',')
#     #     item = {col:data.pop(self.columns.index(col)) for col in ['label', 'seq', 'id']}
#     #     item['emb'] = torch.FloatTensor([float(i) for i in data]) # Add the embedding.
#     #     tf = time.perf_counter()
#     #     # print(f'{f}: Accessed row at index {idx} in {tf - ti}')
#     #     return item

#     def __getitem__(self, idx):
#         '''Loads a specific row from the pandas DataFrame stored at self.data_path.'''
#         f = 'datasets.LowMemoryDataset.__getitem__'

#         # ti = time.perf_counter()
#         data = subprocess.run(f"sed -ne '{idx + 1}p;{idx + 1}q' {self.data_path}", shell=True, check=True, capture_output=True, text=True).stdout.strip().split(',')
#         item = {col:data.pop(self.columns.index(col)) for col in ['label', 'seq', 'id', 'cluster']}
#         item['emb'] = torch.FloatTensor([float(i) for i in data]) # Add the embedding.
#         # tf = time.perf_counter()
#         # print(f'{f}: Accessed row at index {idx} in {tf - ti}')
#         return item

#     def get_labels(self):
#         '''Accessor for the labels associated with the LowMemoryDataset.'''
#         data = pd.read_csv(self.data_path, usecols=['label']) # Only read in the ID values. 
#         # Seems kind of bonkers that I need to ravel this. 
#         return np.ravel(data['label'].values.astype(np.int32)).astype(np.int32)

#     def get_embeddings(self):
#         '''Just in case someone uses a LowMemoryDataset instead of a Dataset by accident. Hopefully gives a more meaningful error message.'''
#         raise RuntimeError('datasets.LowMemoryDataset.get_embeddings: Cannot access all embeddings at once from a LowMemoryDataset. Use a regular Datset.')

#     def get_selenoprotein_indices(self) -> List[int]:
#         '''Accessor for the labels associated with the LowMemoryDataset.'''
#         data = pd.read_csv(self.data_path, usecols=['id']) # Only read in the ID values. 
#         return list(np.where([']' in id for id in data['id'].values])[0])


