'''
This file contains definitions for different Datasets, all of which inherit from the 
torch.utils.data.Dataset class (and are therefore compatible with a Dataloader)
'''

import pandas as pd
import numpy as np
import torch
import h5py
import os 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# It's more convenient to store the embedding information separately from the
# metadata. Or perhaps just write a method to extract it. 

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        '''Initializes a Dataset object, which stores the underlying data.
        
        args:
            - data (pd.DataFrame): The data to be stored in the object. 
        '''
        # Make sure the accession number is set as the index. 
        self.data = data.set_index('id')
        self.length = len(data)

    def __repr__(self):
        pass

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict(orient='records')

    def __len__(self):
        return self.length

    def metadata(self): # Is a DataFrame the best choice of object to return here?
        '''Extract the metadata (i.e. the labels and accession numbers) from the 
        underlying DataFrame. A DataFrame is returned.'''
        return self.data.loc[['label', 'id']]
        # Maybe returning a dictionary would be better?

    def to_csv(self, path):
        '''Write a Dataset object to a CSV file. 

        args:
            - path (str): The path to the CSV file. 
        '''
        df = df.set_index('id')
        df.to_csv(filename, index=True, index_label='id')


class EmbeddingDataset(Dataset):
    
    def __init__(self, data):
        '''Initializes an EmbeddingDataset object.

        args:
            - data (pd.DataFrame): The data to be stored in the object. 
        '''
        # Only want to store the metadata in the DataFrame. 
        super(EmbeddingDataset, self).__init__(data.loc[['label', 'id']])
        
        # Makes sense to store the embeddings as a separate array. 
        self.embs = self.data.drop(columns=['label', 'id'], inplace=False, errors='ignore').values


    def __getitem__(self, idx):
        '''Overrides the Dataset __getitem__ method. This is slightly different, as the
        embedding data is stored in a separate attribute for ease of access.'''
        
        item = super(EmbeddingDataset, self).__getitem__(idx)
        item['embedding'] = self.embds[idx]
        
        return item

    def embeddings(self, return_type='np'):
        '''Extracts the embeddings from the underlying DataFrame.

        args:
            - return_type (str): Indicates the format in which the embeddings
                should be returned. 'np' for numpy, 'pt' for torch.Tensor. 
        '''

        if return_type == 'np':
            return self.embs
        elif return_type == 'pt':
            return torch.Tensor(self.embs)
        else:
            raise ValueError('Return type must be one of: pt, np.')

    def from_h5(path):
        '''Load an EmbeddingDataset object from an H5 file.'''
        file_ = h5py.File(filename)

        ids = file_.keys() # Not sure what this does. 
        data = [file_[key][()] for key in ids] # Not sure what this does either. 
        
        file_.close()

        # Use the embedding data to initialize a DataFrame. 
        data = pd.DataFrame(data)
        data['ids'] = ids

        return EmbeddingDataset(data)

    def from_csv(path):
        '''Load an EmbeddingDataset object from a CSV file.'''
        # Be picky about what is being read into the object. Eventually, we can 
        # add more checks to control what is loaded in (like type restrictions).
        data = pd.read_csv(path, 
            usecols=lambda c : (c in ['label', 'id']) or type(c) == int)

        return EmbeddingDataset(data)

    def to_csv(self, path):
        '''Overwrites the to_csv method defined in the Dataset class, to accomodate
        the storage of the embeddings in a separate attribute.'''

        data = pd.DataFrame(self.embs)
        data = pd.concat([data, self.data], axis=1)
        data = data.set_index('id')

        data.to_csv(path)


class SequenceDataset(Dataset):

    def __init__(self, data):

        super(SequenceDataset, self).__init__(data)

    def sequences(self):
        '''Extract the amino acid sequences from the underlying DataFrame. Returns
        the sequences as a list, for compatibility with tokenizers.'''
        return list(self.data.loc['seq'].values)
 
    def from_csv(path):
        '''Load an EmbeddingDataset object from a CSV file.'''
        # Be picky about what is being read into the object. Eventually, we can 
        # add more checks to control what is loaded in (like type restrictions).
        data = pd.read_csv(path, 
            usecols=lambda c : c in ['label', 'id', 'seq']) 

        return SequenceDataset(data)