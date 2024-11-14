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
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    '''A map-style dataset which  Dataset objects which provide extra functionality for producing sequence embeddings
    and accessing information about selenoprotein content.'''

    def __init__(self, df:pd.DataFrame, n_features:int=1024, half_precision:bool=False, n_classes:int=2):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param n_features: The dimensionality of the stored embeddings. 
        :param half_precision: Whether or not to use half-precision floats. 
        :param n_classes: The number of classes in the labels. 
        '''

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if half_precision else torch.float32
        self.n_features = n_features
        self.n_classes = n_classes

        self.embeddings = torch.from_numpy(df[list(range(n_features))].values).to(self.device).to(self.dtype)
        
        self.labels, self.labels_one_hot_encoded = None, None
        if ('label' in df.columns):
            self.labels =  torch.from_numpy(df['label'].values).type(torch.LongTensor)
            self.labels_one_hot_encoded = one_hot(self.labels, num_classes=self.n_classes).to(self.dtype).to(self.device)

        self.metadata = df[[col for col in df.columns if type(col) == str]] 
        self.ids = df.index.values

        self.length = len(df)
        
    def __len__(self) -> int:
        return self.length

    def scale(self, scaler):
        embeddings = scaler.transform(self.embeddings.cpu().numpy())
        self.embeddings = torch.Tensor(embeddings).to(self.dtype).to(self.device)

    @classmethod
    def from_hdf(cls, path:str, feature_type:str='plm', n_classes:int=2, half_precision:bool=False):
        df = pd.read_hdf(path, key=feature_type)
        n_features = len(df.columns) # Get the number of features. 
        if df.index.name is None:
            df.index.name = 'id' # Forgot to set the index name in some of the files. 
        metadata_df = pd.read_hdf(path, 'metadata')
        df = df.merge(metadata_df, right_index=True, left_index=True, how='inner')
        return cls(df, n_features=n_features, half_precision=half_precision)
    
    def shape(self):
        return self.embeddings.shape

    def sampler(self, p:float=0.99) -> torch.utils.data.WeightedRandomSampler:
        '''Uses labels to generate a WeightedRandomSampler for balancing DataLoader batches.
        
        :param p: The lower bound for the probability that any training instance will be included in the
            final dataset. 
        '''
        labels = self.labels.numpy()

        N = len(labels) # Total number of things in the dataset. 
        n = [(labels == i).sum() for i in range(self.n_classes)]
        # Compute the minimum number of samples such that each training instance will probably be included at least once.
        s = int(max([np.log(1 - p) / np.log(1 - 1 / n_i) for n_i in n])) * self.n_classes
        w = [(N / (n_i * self.n_classes)) for n_i in n]
        
        print(f'Dataset.sampler: {s} samples required for dataset coverage.')
        return torch.utils.data.WeightedRandomSampler([w[i] for i in labels], s, replacement=True)


    def __getitem__(self, idx:int) -> Dict:
        '''Returns an item from the Dataset. Also returns the underlying index for testing purposes.'''
        # embeddings = self.embeddings[:, self.features] # Make sure to filter embeddings by selected features. 
        item = {'embedding':self.embeddings[idx], 'id':self.ids[idx], 'idx':idx}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
            item['label_one_hot_encoded'] = self.labels_one_hot_encoded[idx]
        return item



class BinaryDataset(Dataset):
    categories = {0:'full_length', 1:'truncated_selenoprotein'}
    
    def __init__(self, df:pd.DataFrame, n_features:int=1024, half_precision:bool=False):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param half_precision: Whether or not to use half-precision floats. 
        '''
        assert 'label' in df.columns, 'BinaryDataset.__init__: A BinaryDataset must be labeled.'
        df = df[df.label.isin(list(BinaryDataset.categories.keys()))]
        
        super(BinaryDataset, self).__init__(df, half_precision=half_precision, n_features=n_features, n_classes=2)



class TernaryDataset(Dataset):
    categories = {0:'full_length', 1:'truncated_selenoprotein', 2:'truncated_non_selenoprotein'}
    
    def __init__(self, df:pd.DataFrame, n_features:int=1024, half_precision:bool=False):
        '''Initializes a Dataset from a pandas DataFrame containing embeddings and labels.
        
        :param df: A pandas DataFrame containing the data to store in the Dataset. 
        :param half_precision: Whether or not to use half-precision floats. 
        '''
        assert 'label' in df.columns, 'BinaryDataset.__init__: A TernaryDataset must be labeled.'
        df = df[df.label.isin(list(TernaryDataset.categories.keys()))]
        
        super(TernaryDataset, self).__init__(df, half_precision=half_precision, n_features=n_features, n_classes=3)



def get_dataloader(dataset:Dataset, batch_size:int=16, balance_batches:bool=False) -> DataLoader:
    '''Produce a DataLoader object for each batching of the input Dataset.'''
    if balance_batches:
        return DataLoader(dataset, sampler=dataset.sampler(), batch_size=batch_size)
        # return torch.utils.data.DataLoader(dataset, batch_sampler= BalancedBatchSampler(dataset, batch_size=batch_size))
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)







