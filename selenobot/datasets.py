'''This file contains definitions for a Dataset object, and other associated functions.''' 
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import List, Dict, NoReturn, Iterator
# from sklearn.feature_selection import SelectKBest, f_classif
import re
import subprocess
import time

# TODO: Probably want to split into an EmbeddingDataset and SequenceDataset.

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which  Dataset objects which provide extra functionality for producing sequence embeddings
    and accessing information about selenoprotein content.'''

    def __init__(self, df:pd.DataFrame, embedder=None, n_features:int=None, half_precision:bool=False, device:str='cpu'):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param embedder: The embedder to apply to the amino acid sequences in the DataFrame. 
            If None, it is assumed that the embeddings are already present in the file. 
        '''
        self.seqs = None if 'seq' not in df.columns else df.seq.values
        self.dtype = torch.float16 if half_precision else torch.float32

        # Check to make sure all expected fields are present in the input DataFrame. 
        assert (self.seqs is not None) or (embedder is None), f'dataset.Dataset.__init__: Input DataFrame missing required field seq.'

        self.labels = None if 'label' not in df.columns else torch.from_numpy(df['label'].values).type(self.dtype)
        self.n_features = n_features
        self.embeddings = self._get_embeddings_from_dataframe(df) if embedder is None else embedder(list(self.seqs))
        # self.embeddings = self._select_features(embeddings)
        
        self.type = 'plm' if embedder is None else embedder.type # Type of data contained by the Dataset.
        assert df.index.name == 'gene_id', 'Dataset.__init__: Expecting the DataFrame index to consist of gene ID.'
        self.gene_ids = df.index.values
        self.latent_dim = self.embeddings.shape[-1]

        self.scaler_applied = False
        self.length = len(df)
        self.to_device(device)


    def _get_embeddings_from_dataframe(self, df:pd.DataFrame) -> torch.FloatTensor:
        '''Extract embeddings from an input DataFrame.'''

        # Detect which columns mark an embedding feature (integer column labels). 
        cols = [col for col in df.columns if re.fullmatch(r'\d+', str(col)) is not None]
        embeddings = torch.from_numpy(df[cols].values).to(self.dtype)
        return embeddings
        
    def __len__(self) -> int:
        return self.length

    def apply_scaler(self, scaler, device:str='cpu'):
        # assert not self.scaler_applied, 'Dataset.standardize: Dataset has already been standardized.'
        if not self.scaler_applied:
            self.scaler_applied = True 
            embeddings = scaler.transform(self.embeddings.cpu().numpy())
            # If device is not CPU, make sure to put the dataset back on to the device. 
            self.embeddings = torch.Tensor(embeddings).to(self.dtype).to(device)

    def to_device(self, device):
        '''Put the data stored in the dataset on the device specified on input.'''
        self.device = device
        self.embeddings = self.embeddings.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
    
    def shape(self):
        return self.embeddings.shape

    def __getitem__(self, idx:int) -> Dict:
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        # embeddings = self.embeddings[:, self.features] # Make sure to filter embeddings by selected features. 
        item = {'embedding':self.embeddings[idx], 'gene_id':self.gene_ids[idx], 'idx':idx}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
        return item




def get_dataloader(dataset:Dataset, batch_size:int=16, balance_batches:bool=False) -> torch.utils.data.DataLoader:

    if balance_batches:
        labels = dataset.labels.numpy()
        p = 0.01 # Probability that any given member of a class is not sampled in any batch. 
        n = len(labels)
        n0, n1 = n - np.sum(labels), np.sum(labels)
        s = 2 * int(max(np.log(p) / np.log(1 - 1 / n1), np.log(p) / np.log(1 - 1 / n0))) # Compute the minimum number of samples such that each training instance will probably be included at least once.
        # s = int(np.log(p) / np.log(1 - 1 / n)) # Compute the minimum number of samples such that each training instance will probably be included at least once.
        print(f'get_dataloader: {s} samples required for dataset coverage.')
        w0, w1 = n / (2 * n0), n / (2 * n1) # Compute the probabilities for each class. 
        sampler = torch.utils.data.WeightedRandomSampler([w1 if l == 1 else w0 for l in labels], s, replacement=True)
        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        # return torch.utils.data.DataLoader(dataset, batch_sampler= BalancedBatchSampler(dataset, batch_size=batch_size))
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)




# class BalancedBatchSampler(torch.utils.data.BatchSampler):

#     # TODO: Probably add some checks here, unexpected bahavior might occur if things are evenly divisible by batch size, for example.
#     def __init__(self, data_source:Dataset, batch_size:int=None, random_seed:int=42): 
#         '''Initialize a custom BatchSampler object.'''
#         labels = data_source.labels.to_numpy() # Labels are stored as a Tensor in the Dataset. 
#         classes = np.unique(labels)
#         n_classes = len(classes)
#         max_class = np.argmax([np.sum(labels == c) for c in classes]) # Get the largest class. 
#         max_class_size = np.sum(labels == max_class) # Get the size of the largest class. 
        
#         n_class_per_batch = batch_size // n_classes
#         n_batches = max_class_size // n_class_per_batch

#         batches, n_resampled, n_removed = [], 0, 0
#         random.seed(random_seed) # Seed the random number generator for consistent shuffling.
#         for c in classes:
#             class_idxs = np.where(labels == c).ravel()
#             if c != max_class:
#                 n_resampled += (n_class_per_batch * n_batches) - len(class_idxs)
#                 class_idxs = np.concatenate([np.random.choice(class_idxs, size=(n_class_per_batch * n_batches) - len(class_idxs), replace=True), class_idxs])
#             else:
#                 n_removed = len(class_idxs) - (n_class_per_batch * n_batches)
#                 class_idxs = np.random.choice(class_idxs, size=n_class_per_batch * n_batches, replace=False)
#             np.random.shuffle(class_idxs)
#             batches.append(np.split(class_idxs), n_batches) # Numpy split expects num_batches to be evenly divisible, and will throw an error otherwise. 
    
#         self.batches = np.concatenate(batches, axis=1)

#         # Final check to make sure the number of batches and batch sizes are correct. 
#         assert self.batches.shape == (n_batches, batch_size), f'dataset.BalancedBatchSampler.__init__: Incorrect batch dimensions. Expected {(num_batches, batch_size)}, but dimensions are {self.batches.shape}.'
        
#         print(f'dataset.BalancedBatchSampler.__init__: Resampled {n_resampled} entries and removed {n_removed} entries to generate {n_batches} batches of size {batch_size}.')

#     def __iter__(self) -> Iterator:
#         return iter(self.batches)

#     # Not sure if this should be the number of batches, or the number of elements.
#     def __len__(self) -> int:
#         '''Returns the number of batches.'''
#         return len(self.batches)





