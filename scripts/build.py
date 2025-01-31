import sys, re, os, time
from selenobot.embedders import embed
from selenobot.files import *
import pandas as pd
from typing import NoReturn, Tuple, Dict
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import sklearn
import sklearn.neighbors
import subprocess
import argparse
import logging
import warnings
from selenobot.tools import Clusterer
from selenobot.utils import digitize, groupby, sample, seed

# Label 0: Full-length proteins (both selenoproteins and non-selenoproteins). 
# Label 1: Truncated selenoproteins. 
# Label 2: Truncated non-selenoproteins (mimicking pseudogenes)

TRUNCATED_SYMBOL = '-'
MAX_LENGTH = 1000

warnings.simplefilter('ignore')
seed(42)


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(metadata_df:pd.DataFrame, max_length:int=MAX_LENGTH, **kwargs) -> pd.DataFrame:
    '''''' 
    # There are duplicates here, as there were multiple accessions for the same protein. 
    metadata_df = metadata_df.drop_duplicates('name', keep='first')

    mask = ~(metadata_df.domain == 'Bacteria')
    metadata_df = metadata_df[~mask]
    print(f'clean: Removed {mask.sum()} non-bacterial proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    mask = metadata_df.seq.apply(len) > max_length
    metadata_df = metadata_df[~mask]
    print(f'clean: Removed {mask.sum()} proteins which exceed {max_length} amino acids in length from the DataFrame. {len(metadata_df)} sequences remaining.') 

    # NOTE: Opted to remove all fragmented proteins, not just the C-terminal ones, as I don't trust the annotations of the fragments. 
    mask = ~metadata_df.non_terminal_residue.isnull() # Get a filter for all proteins which have non-terminal residues. 
    metadata_df = metadata_df[~mask]
    print(f'clean: Removed {mask.sum()} fragment proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    return metadata_df



def truncate_sec(metadata_df:pd.DataFrame, **kwargs) -> str:
    '''Truncate the selenoproteins stored in the input file.'''
    metadata_df_truncated = []
    for row in tqdm(metadata_df.to_dict(orient='records'), 'truncate_sec: Truncating selenoproteins...'):
        seq = row['seq'] # Extract the sequence from the row. 
        row['sec_index'] = seq.index('U') # This will raise an exception if no U residue is found.
        row['sec_count'] = seq.count('U') # Store the number of selenoproteins in the original sequence.
        row['truncation_size'] = len(seq) - row['sec_index'] # Store the number of amino acid residues discarded.
        row['truncation_ratio'] = row['truncation_size'] / len(row['seq']) # Store the truncation size as a ratio. 
        row['original_length'] = len(seq)
        row['seq'] = seq[:row['sec_index']] # Get the portion of the sequence prior to the U residue.
        metadata_df_truncated.append(row)
    metadata_df_truncated = pd.DataFrame(metadata_df_truncated, index=[id_ + TRUNCATED_SYMBOL for id_ in metadata_df.index])
    metadata_df_truncated.index.name = 'id'
    return metadata_df_truncated



def split(metadata_df:pd.DataFrame, data_dir:str=None, overwrite:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Divide the uniprot data into training, testing, and validation datasets.'''

    clusterer = Clusterer(tool='mmseqs', name='all_labels', cwd=data_dir)
    metadata_df = clusterer.run(metadata_df, overwrite=overwrite)
    
    n = len(metadata_df) # Get the original number of sequences for checking stuff later. 
    groups = metadata_df['mmseqs_cluster'].values # Extract cluster labels. 
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8)

    idxs, test_idxs = list(gss.split(metadata_df.values, groups=groups))[0]
    test_metadata_df = metadata_df.iloc[test_idxs].copy()
    metadata_df, groups = metadata_df.iloc[idxs].copy(), groups[idxs] # Now working only with the remaining sequence data, not in the test set. 
    
    train_idxs, val_idxs = list(gss.split(metadata_df.values, groups=groups))[0]

    train_metadata_df = metadata_df.iloc[train_idxs].copy()
    val_metadata_df = metadata_df.iloc[val_idxs].copy() 

    return train_metadata_df, test_metadata_df, val_metadata_df


def check(train_metadata_df:pd.DataFrame, test_metadata_df:pd.DataFrame, val_metadata_df:pd.DataFrame):
    
    assert len(np.intersect1d(train_metadata_df.index, val_metadata_df.index)) == 0, 'split: There is leakage between the validation and training datasets.'
    assert len(np.intersect1d(train_metadata_df.index, test_metadata_df.index)) == 0, 'split: There is leakage between the testing and training datasets.'
    assert len(np.intersect1d(test_metadata_df.index, val_metadata_df.index)) == 0, 'split: There is leakage between the validation and testing datasets.'
    

# NOTE: Should I dereplicate the selenoproteins before or after truncation? Seems like after would be a good idea.
def process(path:str, data_dir:str=None, label:int=None, overwrite:bool=False, **kwargs):

    print(f'process: Processing dataset {path}...')

    metadata_df = clean(pd.read_csv(path, index_col=0), **kwargs)

    if label == 1: # Truncate if the dataset is for category 1 or 2. 
        metadata_df = truncate_sec(metadata_df)

    clusterer = Clusterer(name=f'label_{label}', cwd=data_dir, tool='mmseqs')
    metadata_df = clusterer.dereplicate(metadata_df, overwrite=overwrite) # Don't cluster by homology just yet. 
    metadata_df['label'] = label # Add labels to the data marking the category. 

    return metadata_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--ternary', action='store_true')
    parser.add_argument('--quaternary', action='store_true')
    args = parser.parse_args()

    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')

    label_0_metadata_df = process(uniprot_sprot_path, label=0, data_dir=args.data_dir, overwrite=args.overwrite)
    label_1_metadata_df = process(uniprot_sec_path, label=1, data_dir=args.data_dir, overwrite=args.overwrite)
    metadata_df = pd.concat([label_0_metadata_df, label_1_metadata_df])

    train_metadata_df, test_metadata_df, val_metadata_df = split(metadata_df, data_dir=args.data_dir, overwrite=args.overwrite)
    
    metadata = {'train_metadata.csv':train_metadata_df, 'test_metadata.csv':test_metadata_df, 'val_metadata.csv':val_metadata_df}
    for file_name, metadata_df in metadata.items():
        path = os.path.join('.', file_name)
        metadata_df.to_csv(path)
        print(f'Metadata written to {path}')
