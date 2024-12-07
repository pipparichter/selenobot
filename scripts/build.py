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
from selenobot.cdhit import CdHit
from selenobot.utils import digitize, groupby, sample 

# TODO: Figure out why there are two representative columns in the final dataset, which also have different values. Ah, it's 
# probably from the two separate rounds of CD-HIT clustering... should clean this up. 

# Label 0: Full-length proteins (both selenoproteins and non-selenoproteins). 
# Label 1: Truncated selenoproteins. 
# Label 2: Truncated non-selenoproteins. 

warnings.simplefilter('ignore')


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(df:pd.DataFrame, bacteria_only:bool=True, allow_c_terminal_fragments:bool=False, remove_selenoproteins:bool=False, **kwargs) -> pd.DataFrame:
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


def truncate_sec(df:pd.DataFrame, **kwargs) -> str:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all 
    sequences contained in the file contain selenocysteine, labeled as U.'''
    df_truncated = []
    for row in tqdm(df.to_dict(orient='records'), 'truncate: Truncating selenoproteins...'):
        # row['id'] = row['id'] + truncation_label # Modify the row ID to contain a label indicating truncation.
        row['sec_index'] = row['seq'].index('U') # This will raise an exception if no U residue is found.
        row['sec_count'] = row['seq'].count('U') # Store the number of selenoproteins in the original sequence.
        row['truncation_size'] = len(row['seq']) - row['sec_index'] # Store the number of amino acid residues discarded.
        row['seq'] = row['seq'][:row['sec_index']] # Get the portion of the sequence prior to the U residue.
        df_truncated.append(row)
    df_truncated = pd.DataFrame(df_truncated, index=df.index)
    df_truncated.index.name = 'id'
    return df_truncated


def truncate_non_sec(df:pd.DataFrame, sec_seqs:np.ndarray=None, n_bins:int=25, bandwidth:float=0.01, **kwargs) -> pd.DataFrame:
    '''Sub-sample the set of all full-length proteins such that the length distribution matches that of the full-length
    selenoproteins. Then, truncate the sampled sequences so that the truncated length distributions also match.

    :param df: The DataFrame containing the complete set of SwissProt proteins. 
    :param sec_seqs: A Numpy array containing the full-length selenoprotein sequences. 
    :param n_bins: The number of bins to use for producing a length distribution of full-length selenoproteins. 
        This is used when initially down-sampling SwissProt. 
    :param bandwidth: The bandwidth to use for the kernel density estimation, which is used for creating 
        distributions for selecting truncation ratios. 
    :return: A pandas DataFrame containing the sub-sampled and randomly truncated SwissProt proteins. 
    '''
    # Compute the fraction of each selenoprotein which is lost by truncation (sec_truncation_ratios).
    sec_seqs_truncated = np.array([seq.split('U')[0] for seq in sec_seqs])
    sec_lengths = np.array([len(seq) for seq in sec_seqs])
    sec_lengths_truncated = np.array([len(seq) for seq in sec_seqs_truncated]) 
    sec_truncation_ratios = (sec_lengths - sec_lengths_truncated) / sec_lengths

    # Group the lengths of the full-length selenoproteins into n_bins bins
    hist, bin_edges = np.histogram(sec_lengths, bins=n_bins)
    bin_labels, bin_names = digitize(sec_lengths, bin_edges)

    # Sample from the DataFrame of full-length SwissProt proteins, ensuring that the length distribution matches
    # that of the full-length selenoproteins. 
    _, idxs = sample(df.seq.apply(len).values, hist, bin_edges)
    # Assign each of the sampled proteins a bin label, where the bins are the same as those of the full-length selenoprotein
    # length distribution (from above).
    df = df.iloc[idxs].copy()
    df['bin_label'], _ = digitize(df.seq.apply(len).values, bin_edges)

    # Generate a continuous distribution for truncation ratios for each length bin. This is necessary because of 
    # heteroscedacity: the variance in the truncation ratios of short sequences is much higher than that of long
    # sequences, so need to generate different distributions for different length categories. 
    kdes = dict()
    for bin_label, bin_values in groupby(sec_truncation_ratios, bin_labels).items():
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth) 
        kde.fit(bin_values.reshape(-1, 1))
        kdes[bin_label] = kde

    # Use the KDE to sample truncation ratios for each length bin, and apply the truncation to the full-length sequence. 
    df_truncated = []
    for bin_label, bin_df in df.groupby('bin_label'):
        bin_df['truncation_size'] = kdes[bin_label].sample(n_samples=len(bin_df)).ravel() * bin_df.seq.apply(len).values
        bin_df['seq'] = bin_df.apply(lambda row : row.seq[:-int(row.truncation_size)], axis=1)
        df_truncated.append(bin_df)

    df_truncated = pd.concat(df_truncated).drop(columns=['bin_label'])
    df_truncated.index.name = 'id'
    print('df_truncated')
    print(df_truncated)
    return df_truncated


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
def process(path:str, datasets:Dict[str, List[pd.DataFrame]], data_dir:str=None, label:int=0,  **kwargs):

    print(f'process: Processing dataset {path}...')

    df = clean(pd.read_csv(path, index_col=0), **kwargs)

    if label == 1: # Truncate if the dataset is for category 1 or 2. 
        df = truncate_sec(df)
    elif label == 2:
        df = truncate_non_sec(df, **kwargs)

    name = os.path.basename(path).replace('.csv', '')
    df = CdHit(df, name=name, cwd=data_dir).run(overwrite=False)

    df['label'] = label # Add labels to the data marking the category. 
    # Decided to split each data group independently to avoid the mixed clusters. 
    train_df, test_df, val_df = split(df) 

    # Append the split DataFrames to the lists for concatenation later on. 
    datasets['train.h5'].append(train_df)
    datasets['test.h5'].append(test_df)
    datasets['val.h5'].append(val_df)

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--print-stats', action='store_true')
    args = parser.parse_args()

    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')

    datasets = {'train.h5':[], 'test.h5':[], 'val.h5':[]}

    process(uniprot_sprot_path, datasets, label=0, data_dir=args.data_dir)
    sec_df = process(uniprot_sec_path, datasets, label=1, data_dir=args.data_dir, allow_c_terminal_fragments=True)
    process(uniprot_sprot_path, datasets, label=2, data_dir=args.data_dir, allow_c_terminal_fragments=True, remove_selenoproteins=True, sec_seqs=sec_df.seq.values)

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
