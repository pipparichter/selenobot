'''This file contains definitions for a Dataset object, and other associated functions.''' 
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from typing import List, Dict, NoReturn, Iterator
from sklearn.feature_selection import SelectKBest, f_classif
import re
import subprocess
import time

# TODO: Probably want to split into an EmbeddingDataset and SequenceDataset.

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseDataset(torch.utils.data.Dataset):
    pass

class SequenceDataset(BaseDataset):
    pass

class EmbeddingDataset(BaseDataset):
    pass


class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which  Dataset objects which provide extra functionality for producing sequence embeddings
    and accessing information about selenoprotein content.'''
    
    def __init__(self, df:pd.DataFrame, embedder=None, n_features:int=None):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param embedder: The embedder to apply to the amino acid sequences in the DataFrame. 
            If None, it is assumed that the embeddings are already present in the file. 
        '''
        self.seqs = None if 'seq' not in df.columns else df.seq.values

        # Check to make sure all expected fields are present in the input DataFrame. 
        assert (self.seqs is not None) or (embedder is None), f'dataset.Dataset.__init__: Input DataFrame missing required field seq.'

        self.embeddings = self._get_embeddings_from_dataframe(df) if embedder is None else embedder(list(self.seqs))
        self.type = 'plm' if embedder is None else embedder.type # Type of data contained by the Dataset.
        self.labels = None if 'label' not in df.columns else torch.from_numpy(df['label'].values).type(torch.float32)
        self.gene_ids = df.index.values
        self.latent_dim = self.embeddings.shape[-1]
        self.n_features = n_features
        self.features, self.feature_scores = self._select_features()

        self.standardized = False

        self.length = len(df)

    def _select_features(self):
        if self.n_features is None:
            return np.arange(self.embeddings.shape[1]), None
        else:
            # NOTE: Should I be using t-test, as there are technically only two categories?
            kbest = SelectKBest(f_classif, k='all') # Use ANOVA to select the best features. 
            kbest.fit(self.embeddings, self.labels)
            # Scores are the ANOVA F-statistic, which is the ratio of the between-group variability to the within-group variability.
            # The larger this value, the "better" the groups, in a sense. 
            feature_scores = kbest.scores_
            idxs = np.argsort(feature_scores) # argsort sorts in ascending order.
            return idxs[-self.n_features:], feature_scores # Grab the n_features features with the highest F-scores. 

    def _get_embeddings_from_dataframe(self, df:pd.DataFrame) -> torch.FloatTensor:
        '''Extract embeddings from an input DataFrame.'''
        # Detect which columns mark an embedding feature. 
        cols = [col for col in df.columns if re.fullmatch(r'\d+', col) is not None]
        embeddings = torch.from_numpy(df[cols].values).to(torch.float32)
        return embeddings

    def get_embeddings(self):
        '''Return the embeddings stored in the object, filtering for significant features.'''
        return self.embeddings[:, self.features]
        
    def __len__(self) -> int:
        return self.length

    def standardize(self, scaler):
        assert not self.standardized, 'Dataset.standardize: Dataset has already been standardized.'
        self.standardized = True 
        embeddings = scaler.transform(self.embeddings)
        self.embeddings = torch.Tensor(embeddings).to(torch.float32)

    def to_device(self, device):
        '''Put the data stored in the dataset on the device specified on input.'''
        self.embeddings = self.embeddings.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
    
    def shape(self):
        return (len(self), len(self.features))

    def __getitem__(self, idx:int) -> Dict:
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        embeddings = self.embeddings[:, self.features] # Make sure to filter embeddings by selected features. 
        item = {'embedding':embeddings[idx], 'gene_id':self.gene_ids[idx], 'idx':idx}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
        return item

    def get_selenoprotein_indices(self) -> List[int]:
        '''Obtains the indices of selenoproteins in the Dataset.'''
        return list(np.where([']' in i for i in self.gene_ids])[0])


# A BatchSampler should have an __iter__ method which returns the indices of the next batch once called.
class BalancedBatchSampler(torch.utils.data.BatchSampler):
    '''A Sampler object which ensures that each batch has a similar proportion of selenoproteins to non-selenoproteins.
    This class is used in conjunction with PyTorch DataLoaders.'''

    # TODO: Probably add some checks here, unexpected bahavior might occur if things are evenly divisible by batch size, for example.
    def __init__(self, data_source:Dataset, batch_size:int=None, random_seed:int=42): 
        '''Initialize a custom BatchSampler object.
        
        :param data_source: A Dataset object containing the data to sample into batches. 
        :param batch_size: The size of the batches. 
        :param random_seed: The value to seed the random number generator. 
        '''
        r = 0.5 # The fraction of truncated selenoproteins to include in each batch. 
        random.seed(random_seed) # Seed the random number generator for consistent shuffling.

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
        random_seed:int=42, 
        num_workers:int=0) -> torch.utils.data.DataLoader:
    '''Create a DataLoader from a CSV file containing sequence and/or PLM embedding data.
    
    :param dataset: The Dataset used to generate the DataLoader. 
    :param batch_size: The size of the batches which the training data will be split into. 
    :return: A pytorch DataLoader object. 
    '''
    batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size, random_seed=random_seed)
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)




