'''This file contains definitions for different Datasets, all of which inherit from the 
torch.utils.data.Dataset class (and are therefore compatible with a Dataloader)'''

import pandas as pd
import numpy as np
import torch
import h5py
import os
import torch
from torch.utils.data import DataLoader
from utils import pd_from_fasta
import random
from tqdm import tqdm

# Importing my own modules. 
import embedders
import utils

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(
        path:str, 
        batch_size:int=1024, 
        balance:bool=False, 
        embedder:str=None,
        verbose:bool=True) -> DataLoader:
    '''Create a DataLoader from a CSV of embedding data.
    
    args:
        - path: The path to the CSV file where the data is stored. 
        - batch_size: Batch size of the DataLoader. If None, no batches are used. 
        - balance: Whether or not to ensure that each batch is balanced for positive or negative cases.
        - embedder: A string specifying the embedder to apply to the data. 
        - verbose: Whether or not to print various updates during runtime.  
    '''
    if embedder is None: # If no embedder is specified, just load in data from the file. 
        dataset = Dataset.from_csv(path)
    else:
        if embedder == 'aac':
            embedder = embedders.AacEmbedder()
        elif embedder == 'length':
            embedder = embedders.LengthEmbedder()
        elif embedder == 'plm':
            raise Exception('TODO')
        else:
            raise Exception('dataset.get_dataloader: Invalid embedding method given.')
        
        # It makes more sense to pass in a path to the __.csv files, which have embedding information, as well as the sequences. 
        seqs = np.ravel(pd.read_csv(path, usecols=['seq']).values)
        ids = np.ravel(pd.read_csv(path, usecols=['id']).values)
        labels = np.ravel(pd.read_csv(path, usecols=['label']).values)

        # Embed the sequence data. 
        embeddings = embedder(seqs)

        # Collect all the information into a pandas DataFrame. 
        data = pd.DataFrame(embeddings)
        # Add other information to the DataFrame. 
        data['seq'] = seqs
        data['id'] = ids
        data['label'] = labels
        data = data.set_index('id')

        dataset = Dataset(data)
     
    # If the batch size is None, load the entire Dataset at once. 
    batch_size = len(dataset) if batch_size is None else batch_size

    if balance:
        # Providing batch_sampler will override batch_size, shuffle, sampler, and drop_last altogether. 
        # It is meant to define exactly the batch elements and their content.
        batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size, percent_positive_instances=0.25)
        return DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        return DataLoader(dataset, shuffle=True, batch_size=batch_size)


# Because this implements __getitem__, it is a map-style dataset. 
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        '''Initializes an EmbeddingDataset from a pandas DataFrame containing
        embeddings and labels.'''

        # Makes sense to store the embeddings as a separate array. 
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

    def from_csv(path):
        '''Load an Dataset object from a CSV file.'''
        data = pd.read_csv(path, index_col='id')
        assert 'seq' in data.columns, 'dataset.Dataset.from_csv: Column seq is not in data file.'
        assert 'label' in data.columns, 'dataset.Dataset.from_csv: Column label is not in data file.'
        return Dataset(data)

    def get_positive_idxs(self, shuffle=True) -> list:
        idxs = list(np.where([']' in id for id in self.ids_])[0])
        if shuffle:
            random.shuffle(idxs)
        return idxs

    def get_negative_idxs(self, shuffle=True) ->list:
        idxs = list(np.where([']' not in id for id in self.ids_])[0])
        if shuffle:
            random.shuffle(idxs)
        return idxs


# A BatchSampler should have an __iter__ method which returns the indices of the next batch once called.
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    '''A Sampler object which ensures that each batch has a similar proportion of selenoproteins to non-selenoproteins.'''

    def __init__(self, data_source, batch_size=None,  percent_positive_instances=0.25):
        

        num_pos_per_batch = int(batch_size * percent_positive_instances)
        num_neg_per_batch = batch_size - num_pos_per_batch

        pos_idxs, neg_idxs = data_source.get_positive_idxs(shuffle=True), data_source.get_negative_idxs(shuffle=True)
        num_pos, num_neg = len(pos_idxs), len(neg_idxs)

        pos_info = {'unsampled':set(pos_idxs), 'sampled':set(), 'num_per_batch':num_pos_per_batch, 'num_resampled':0, 'num':num_pos}
        neg_info = {'unsampled':set(neg_idxs), 'sampled':set(), 'num_per_batch':num_neg_per_batch, 'num_resampled':0, 'num':num_neg}

        # Going to make the assumption that there are more negative than positive instances. 
        assert num_pos < num_neg, 'classifiers.BalancedBatchSampler.__init__: Expect the number of negative instances to be greater than the number of positive instances.'
        assert batch_size <= len(data_source), f'classifiers.BalancedBatchSampler.__init__: Specified batch size {batch_size} must not exceed the size of the dataset.'
   
        # Want to select a number of batches such that the entirety of both classes is covered. 
        # (assumed to be the group of negative instances) is covered. 
        self.num_batches = max(num_neg // num_neg_per_batch, num_pos // num_pos_per_batch) + 1
        # NOTE: This might sometimes oversample from the dataset, but shouldn't be a big deal. 

        self.batch_size = batch_size

        self.batches = self.get_batches(pos_info, neg_info)

    def get_batches(self, pos_info, neg_info, verbose=True):

        batches = []
        for i in tqdm(range(self.num_batches), desc='dataset.BalancedBatchSampler.get_batches', leave=True):
            
            batch = []

            # Accounting for the instance in which 
            for info in [pos_info, neg_info]:

                if len(info['unsampled']) < info['num_per_batch']:
                    # Get the number of indices which are not covered by the unsampled group.
                    n = info['num_per_batch'] - len(info['unsampled'])
                    # Re-sampled from the sampled group to accommodate the deficiency. 
                    info['unsampled'] = info['unsampled'].union(random.sample(list(info['sampled']), n))
                    info['num_resampled'] += n # Keep track of the number of resampled elements. 

                # NOTE: random.sample is without replacement by default. However, it spits out a list, so need to conert back to a set. 
                sample = set(random.sample(list(info['unsampled']), info['num_per_batch']))
                # Update the dictionary by moving the sampled elements to the sampled list. 
                info['sampled'] = info['sampled'].union(sample - info['sampled']) # Make sure to not add any index which is being re-sampled. Should be handled, because objects are sets. 
                info['unsampled'] = info['unsampled'] - sample
                # Add the sample to the batch. 
                batch += list(sample)

                # Make sure the total number of instances remains fixed. 
                assert info['num'] == (len(info['sampled']) + len(info['unsampled'])), f'classifiers.BalancedBatchSampler.get_batches: Number of positive indices is inconsistent.'

            # Make sure to shuffle the batch. Othereise, all the positive cases come before the negative, and the curve looks weird. 
            random.shuffle(batch)
            assert len(batch) == self.batch_size, f'classifiers.BalancedBatchSampler.get_batches: Batch size {len(batch)} fores not match specified {self.batch_size}.'
            batches.append(batch)
        
        if verbose: print(f"classifiers.BalancedBatchSampler.get_batches: {pos_info['num_resampled']} positive instances resampled.")
        if verbose: print(f"classifiers.BalancedBatchSampler.get_batches: {neg_info['num_resampled']} negative instances resampled.")

        return batches

    def __iter__(self):
        return iter(self.batches)

    # Not sure if this should be the number of batches, or the number of elements.
    def __len__(self):
        return self.num_batches

    # def from_csv(path, ids=None):
    #     '''Load an EmbeddingDataset object from a CSV file.'''

    #     # Load in all the sequences to a DataFrame. 
    #     data = pd.read_csv(path, index_col='id')

    #     if ids is not None:
    #         # Use the specified IDs to filter the data. 
    #         data = data[np.isin(data.index, ids)]

    #     return SequenceDataset(data)