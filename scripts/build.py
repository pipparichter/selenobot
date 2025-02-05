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


label_names = dict()
label_names[0] = 'full-length'
label_names[1] = 'truncated selenoprotein'
label_names[2] = 'short full-length'

warnings.simplefilter('ignore')
seed(42)

# Before, I had been relying on CD-HIT to filter out short sequences, but it seems as though MMSeqs does not have a lower length limit. 
# Some of the truncated selenoproteins are absurdly short, only one amino acid in length, which I think could mess up the signal. There's
# no way there's any meaningful information from a PLM in the embedding of a single amino acid.

# This makes me a little wary of the idea that there will be a stronger signal in the last few amino acids of the sequence, as compared
# to the entire mean-pooled sequence... seems to point to the idea that the real signal is downstream of the opal codon. But I will try it anyway. 

# Additionally, Prodigal has a default minimum sequence length of 90 bp (which it seems that GTDB uses). This means that using the GTDB predicted 
# ORFs might be a bad idea, as it might be missing many truncated selenoproteins altogether. I think the right thing to do is to get all of the
# GTDB genomes and re-run Prodigal with a different minimum length. 


# NOTE: Prodigal's minimum length is not customizable, so will need to use a different tool (Pyrodigal). 

MAX_LENGTH = 1000 # Mostly a limit to make embedding tractable. 
MIN_LENGTH = 10

def clean(metadata_df:pd.DataFrame, min_length:int=MIN_LENGTH, max_length:int=MAX_LENGTH) -> pd.DataFrame:
    '''''' 
    # There are duplicates here, as there were multiple accessions for the same protein. 
    metadata_df = metadata_df.drop_duplicates('name', keep='first')

    mask = (metadata_df.seq.apply(len) < max_length) & (metadata_df.seq.apply(len) >= min_length) 
    metadata_df = metadata_df[mask]
    print(f'clean: Removed {mask.sum()} proteins with lengths out of the range {min_length} to {max_length}.')

    mask = ~(metadata_df.domain == 'Bacteria')
    metadata_df = metadata_df[~mask]
    print(f'clean: Removed {mask.sum()} non-bacterial proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    # NOTE: Opted to remove all fragmented proteins, not just the C-terminal ones, as I don't trust the annotations of the fragments. 
    mask = ~metadata_df.non_terminal_residue.isnull() # Get a filter for all proteins which have non-terminal residues. 
    metadata_df = metadata_df[~mask]
    print(f'clean: Removed {mask.sum()} fragment proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    return metadata_df



def truncate_sec(metadata_df:pd.DataFrame, trunc_symbol:str='-') -> str:
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
    metadata_df_truncated = pd.DataFrame(metadata_df_truncated, index=[id_ + trunc_symbol for id_ in metadata_df.index])
    metadata_df_truncated.index.name = 'id'
    return metadata_df_truncated



def split(metadata_df:pd.DataFrame, overwrite:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Divide the uniprot data into training, testing, and validation datasets.'''

    clusterer = Clusterer(tool='mmseqs', name=f'{mode}_all', cwd=data_dir)
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
    

def build(path:str=None, label:int=None, overwrite:bool=False, **kwargs):
    metadata_df = pd.read_csv(path, index_col=0)

    # Make sure to truncate BEFORE filtering by length.
    if label == 1: # Truncate if the dataset is for category 1. 
        metadata_df = truncate_sec(metadata_df)

    print(f'build: Processing data for group "{label_names[label]}"...')
    metadata_df = clean(metadata_df, **kwargs)
    
    clusterer = Clusterer(name=f'{mode}_{label}', cwd=data_dir, tool='mmseqs')
    metadata_df = clusterer.dereplicate(metadata_df, overwrite=overwrite) # Don't cluster by homology just yet. 
    metadata_df['label'] = label # Add labels to the data marking the category. 

    return metadata_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n-classes', default=2, type=int)
    args = parser.parse_args()

    global mode 
    mode = f'{args.n_classes}c'

    global data_dir
    data_dir = args.data_dir

    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')

    kwargs = dict()
    kwargs[0] = {'path':uniprot_sprot_path, 'min_length':200 if (args.n_classes == 3) else MIN_LENGTH, 'max_length':MAX_LENGTH}
    kwargs[1] = {'path':uniprot_sec_path, 'min_length':MIN_LENGTH, 'max_length':MAX_LENGTH}
    kwargs[2] = {'path':uniprot_sprot_path, 'min_length':MIN_LENGTH, 'max_length':200}

    metadata_df = pd.concat([build(label=label, overwrite=args.overwrite, **kwargs.get(label)) for label in range(args.n_classes)])

    train_metadata_df, test_metadata_df, val_metadata_df = split(metadata_df, overwrite=args.overwrite)
    
    metadata = {f'{mode}_metadata_train.csv':train_metadata_df, f'{mode}_metadata_test.csv':test_metadata_df, f'{mode}_metadata_val.csv':val_metadata_df}
    for file_name, metadata_df in metadata.items():
        path = os.path.join(args.data_dir, file_name)
        metadata_df.to_csv(path)
        print(f'Metadata written to {path}')
