import sys, re, os, time
from selenobot.embedders import embed
from selenobot.files import *
import pandas as pd
from typing import NoReturn, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import subprocess
import argparse
import logging
import warnings
from selenobot.cdhit import CdHit


warnings.simplefilter('ignore')


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(df:pd.DataFrame, bacteria_only:bool=True, allow_c_terminal_fragments:bool=False, remove_selenoproteins:bool=True) -> str:
    '''''' 
    # There are duplicates here, as there were multiple accessions for the same protein. 
    df = df.drop_duplicates('name', keep='first')

    if bacteria_only:
        non_bacterial = ~(df.domain == 'Bacteria')
        df = df[~non_bacterial]
        print(f'clean: Removed {non_bacterial.sum()} non-bacterial proteins from the DataFrame. {len(df)} sequences remaining.') 

    if remove_selenoproteins: # TODO: Add a check to make sure this catches everything... 
        selenoprotein = df.seq.str.contains('U')
        df = df[~selenoprotein]
        print(f'clean: Removed {selenoprotein.sum()} selenoproteins from the DataFrame. {len(df)} sequences remaining.') 

    # Remove fragmented proteins from the DataFrame.
    # This function checks to see if there is a non-terminal residue at the beginning of the protein (i.e. N-terminal). 
    # If there is not, the fragment is C-terminal. 
    is_c_terminal_fragment = lambda pos : ('1' not in pos.split(',')) if (type(pos) == str) else False
    fragmented = ~df.non_terminal_residue.isnull() # Get a filter for all proteins which have non-terminal residues. 
    if allow_c_terminal_fragments:
        # If specified, require that the non-terminal residue must be on the N-terminus (beginning) for removal. 
        fragmented = np.logical_and(fragmented, ~df.non_terminal_residue.apply(is_c_terminal_fragment) )
    df = df[~fragmented]
    print(f'clean: Removed {fragmented.sum()} fragment proteins from the DataFrame. {len(df)} sequences remaining.') 

    return df


def truncate(df:pd.DataFrame) -> str:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all 
    sequences contained in the file contain selenocysteine, labeled as U.'''
    df_truncated = []
    for row in tqdm(df.to_dict(orient='records'), 'truncate: Truncating selenoproteins...'):
        # row['id'] = row['id'] + truncation_label # Modify the row ID to contain a label indicating truncation.
        row['sec_index'] = row['seq'].index('U') # This will raise an exception if no U residue is found.
        row['sec_count'] = row['seq'].count('U') # Store the number of selenoproteins in the original sequence.
        row['trunc'] = len(row['seq']) - row['sec_index'] # Store the number of amino acid residues discarded.
        row['seq'] = row['seq'][:row['sec_index']] # Get the portion of the sequence prior to the U residue.
        df_truncated.append(row)
    df_truncated = pd.DataFrame(df_truncated, index=df.index)
    df_truncated.index.name = 'id'
    return df_truncated

# TODO: I think I want to make sure that the cluster size in each split dataset are roughly equivalent, as 
# more clusters relative to the total size of the cluster would mean more statistical "power" within the dataset. 
# I would want them to be relatively equal. 

def split(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Divide the uniprot data into training, testing, and validation datasets. The split is cluster-aware, and ensures that
    no CD-HIT-generated cluster is split between datasets. The data is first divided into a training set (80 percent) and test
    set (20 percent), and the training data is further divided into a training (80 percent) and validation (20 percent) set.'''

    groups = df['cluster'].values # Extract cluster labels. 
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8)

    idxs, test_idxs = list(gss.split(df.values, groups=groups))[0]

    test_df = df.iloc[test_idxs].copy()
    # Now working only with the remaining sequence data, not in the test set. 
    df, groups = df.iloc[idxs].copy(), groups[idxs]

    train_idxs, val_idxs = list(gss.split(df.values, groups=groups))[0]

    train_df = df.iloc[train_idxs].copy()
    val_df = df.iloc[val_idxs].copy() 
    # print(len(train_df), len(test_df), len(val_df))
    
    return train_df, test_df, val_df


def stats(df:pd.DataFrame, name:str=None):
    print(f'\nstats: Information for {name}')

    size = len(df)
    fraction_selenoproteins = np.round(df.label.sum() / size, 2)
    mean_cluster_size = np.round(df.groupby('cluster').apply(len).mean(), 2)
    n_clusters = len(df.cluster.unique()) 

    print(f'stats: Dataset size:', size)
    print(f'stats: Fraction of selenoproteins:', fraction_selenoproteins)
    print(f'stats: Mean cluster size (80%):', mean_cluster_size)
    print(f'stats: Number of clusters (80%):', n_clusters)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    # parser.add_argument('--bacteria-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print-stats', action='store_true')
    args = parser.parse_args()

    # NOTE: Should I dereplicate the selenoproteins before or after truncation? Seems like after would be a good idea. 

    # Define all the relevant file paths.
    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    print(f'Processing {uniprot_sprot_path}...')
    
    uniprot_sprot_df = clean(pd.read_csv(uniprot_sprot_path, index_col=0), bacteria_only=True, remove_selenoproteins=True, allow_c_terminal_fragments=False)
    uniprot_sprot_df = CdHit(uniprot_sprot_df, name='uniprot_sprot', cwd=args.data_dir).run()

    
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')
    print(f'\nProcessing {uniprot_sec_path}...')
    uniprot_sec_df = clean(pd.read_csv(uniprot_sec_path, index_col=0), bacteria_only=True, allow_c_terminal_fragments=True, remove_selenoproteins=False)
    uniprot_sec_df = truncate(uniprot_sec_df) # Truncate the selenoproteins. 
    uniprot_sec_df = CdHit(uniprot_sec_df, name='uniprot_sec', cwd=args.data_dir).run()

    
    uniprot_sec_df['label'] = 1 # Add labels marking the truncated selenoproteins. 

    # Add labels to the data, indicating whether or not each sequence is a truncated selenoprotein. 
    uniprot_sprot_df['label'] = 0

    # Decided to split each data group independently to avoid the mixed clusters. 
    uniprot_sprot_train_df, uniprot_sprot_test_df, uniprot_sprot_val_df = split(uniprot_sprot_df)
    uniprot_sec_train_df, uniprot_sec_test_df, uniprot_sec_val_df = split(uniprot_sec_df)

    train_df = pd.concat([uniprot_sprot_train_df, uniprot_sec_train_df])
    test_df = pd.concat([uniprot_sprot_test_df, uniprot_sec_test_df])
    val_df = pd.concat([uniprot_sprot_val_df, uniprot_sec_val_df])

    if args.print_stats:
        stats(train_df, name='train_df')
        stats(test_df, name='train_df')
        stats(val_df, name='val_df')
        print()

    # for file_name, df in zip(['train.h5', 'test.h5', 'val.h5'], [train_df, test_df, val_df]):
    #     embed(df, path=os.path.join(args.data_dir, file_name))
    #     print(f'Dataset saved to {os.path.join(args.data_dir, filename)}')
