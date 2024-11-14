import sys, re, os, time
from selenobot.embedders import embed
from selenobot.files import *
import pandas as pd
from typing import NoReturn, Tuple, Dict
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import subprocess
import argparse
import logging
import warnings
from selenobot.cdhit import CdHit

warnings.simplefilter('ignore')

# NOTE: Need to think about how to grab sequences for the new class (truncated non-selenoproteins). 
# I could potentially use the sequences which are fragmented at the C-terminus (i.e. the end), and just truncate them further. 
# I could also just subset the negative cases. How many should I grab? Would it be reasonable to make the number of 
# truncated non-selenoproteins equal to the number of truncated selenoproteins? Seems like this might be a good idea?
# Also need to think about how to pick truncation lengths. 


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(df:pd.DataFrame, bacteria_only:bool=True, allow_c_terminal_fragments:bool=False, remove_selenoproteins:bool=False) -> pd.DataFrame:
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


def truncate(df:pd.DataFrame, mode:int=1) -> str:
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



# NOTE: Should I dereplicate the selenoproteins before or after truncation? Seems like after would be a good idea.
def process(path:str, datasets:Dict[str, List[pd.DataFrame]], data_dir:str=None, label:int=0, **kwargs):

    print(f'process: Processing dataset {path}...')

    df = clean(pd.read_csv(path, index_col=0), **kwargs)

    if label > 0: # Truncate if the dataset is for category 1 or 2. 
        df = truncate(df, mode=label)

    name = os.path.basename(path).replace('.csv', '')
    df = CdHit(df, name=name, cwd=data_dir).run(overwrite=False)

    df['label'] = label # Add labels to the data marking the category. 
    # Decided to split each data group independently to avoid the mixed clusters. 
    train_df, test_df, val_df = split(df) 

    # Append the split DataFrames to the lists for concatenation later on. 
    datasets['train.h5'].append(train_df)
    datasets['test.h5'].append(test_df)
    datasets['val.h5'].append(val_df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--categories', default=[0, 1, 2], action='store', type=int, nargs='*')
    parser.add_argument('--print-stats', action='store_true')
    args = parser.parse_args()
    
    datasets = {'train.h5':[], 'test.h5':[], 'val.h5':[]}
    source_files = {0:'uniprot_sprot.csv', 1:'uniprot_sec.csv', 2:'uniprot_trembl.csv'}

    # Define the keyword arguments for the clean function for each category. 
    kwargs = dict()
    kwargs[0] = {}
    kwargs[1] = {'allow_c_terminal_fragments':True}
    kwargs[2] = {'allow_c_terminal_fragments':True, 'remove_selenoproteins':True}
    
    os.chdir(args.data_dir) # Set the current working directory to avoid having to use full paths. 

    for category in args.categories:
        path = os.path.join(args.data_dir, source_files[category])
        process(path, datasets, label=category, data_dir=args.data_dir, **kwargs[category])

    # Concatenate the accumulated datasets. 
    datasets = {name:pd.concat(dfs) for name, dfs in datasets.items()}

    if args.print_stats:
        for file_name, df in datasets.items():
            stats(df, name=file_name)
        print()

    # NOTE: I want to be able to add to exsting HDF files. 
    for file_name, df in datasets.items():
        path = os.path.join(args.data_dir, file_name)
        embed(df, path=path, append=args.append)
