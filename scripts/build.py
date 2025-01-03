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

# TODO: Bug related to the fact that the cluster labels are not unique, as I use MMSeqs multiple separate times. 
# TODO: Why did I think it was a problem to have mixed clusters? I think there is a problem where truncated selenoproteins
# strongly resemble their full-length counterparts. I really think I should be de-replicating and clustering after creating
# the entire dataset, as long as I still end up with roughly even distributions. 

# Label 0: Full-length proteins (both selenoproteins and non-selenoproteins). 
# Label 1: Truncated selenoproteins. 
# Label 2: Truncated non-selenoproteins. 

TRUNCATED_SYMBOL = '-'

warnings.simplefilter('ignore')
seed(42)


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(df:pd.DataFrame, bacteria_only:bool=True, allow_c_terminal_fragments:bool=False, remove_selenoproteins:bool=False, max_length:int=2000, **kwargs) -> pd.DataFrame:
    '''''' 
    # There are duplicates here, as there were multiple accessions for the same protein. 
    df = df.drop_duplicates('name', keep='first')

    if bacteria_only:
        non_bacterial = ~(df.domain == 'Bacteria')
        df = df[~non_bacterial]
        print(f'clean: Removed {non_bacterial.sum()} non-bacterial proteins from the DataFrame. {len(df)} sequences remaining.') 

    if remove_selenoproteins: # TODO: Add a check to make sure this catches everything... 
        is_a_selenoprotein = df.seq.str.contains('U')
        df = df[~is_a_selenoprotein]
        print(f'clean: Removed {is_a_selenoprotein.sum()} selenoproteins from the DataFrame. {len(df)} sequences remaining.') 

    if max_length is not None:
        exceeds_max_length = df.seq.apply(len) > max_length
        df = df[~exceeds_max_length]
        print(f'clean: Removed {exceeds_max_length.sum()} proteins which exceed {max_length} amino acids in length from the DataFrame. {len(df)} sequences remaining.') 

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
    for row in tqdm(df.to_dict(orient='records'), 'truncate_sec: Truncating selenoproteins...'):
        seq = row['seq'] # Extract the sequence from the row. 
        row['sec_index'] = seq.index('U') # This will raise an exception if no U residue is found.
        row['sec_count'] = seq.count('U') # Store the number of selenoproteins in the original sequence.
        row['truncation_size'] = len(seq) - row['sec_index'] # Store the number of amino acid residues discarded.
        row['truncation_ratio'] = row['truncation_size'] / len(row['seq']) # Store the truncation size as a ratio. 
        row['original_length'] = len(seq)
        row['seq'] = seq[:row['sec_index']] # Get the portion of the sequence prior to the U residue.
        df_truncated.append(row)
    df_truncated = pd.DataFrame(df_truncated, index=[id_ + TRUNCATED_SYMBOL for id_ in df.index])
    df_truncated.index.name = 'id'
    return df_truncated


def truncate_non_sec(df:pd.DataFrame, label_1_df:pd.DataFrame=None, n_bins:int=10, bandwidth:float=0.01, **kwargs) -> pd.DataFrame:
    '''Sub-sample the set of all full-length proteins such that the length distribution matches that of the full-length
    selenoproteins. Then, truncate the sampled sequences so that the truncated length distributions also match.

    :param df: The DataFrame containing the complete set of SwissProt proteins. 
    :param sec_df: 
    :param n_bins: The number of bins to use for producing a length distribution of full-length selenoproteins. 
        This is used when initially down-sampling SwissProt. 
    :param bandwidth: The bandwidth to use for the kernel density estimation, which is used for creating 
        distributions for selecting truncation ratios. 
    :return: A pandas DataFrame containing the sub-sampled and randomly truncated SwissProt proteins. 
    '''
    sec_lengths = label_1_df.original_length.values
    sec_truncation_ratios = label_1_df.truncation_ratio.values

    # Group the lengths of the full-length selenoproteins into n_bins bins
    print(f'truncate_non_sec: Generating a selenoprotein length distribution with {n_bins} bins.')
    hist, bin_edges = np.histogram(sec_lengths, bins=n_bins)
    bin_labels, bin_names = digitize(sec_lengths, bin_edges)

    # Sample from the DataFrame of full-length SwissProt proteins, ensuring that the length distribution matches
    # that of the full-length selenoproteins. 
    _, idxs = sample(df.seq.apply(len).values, hist, bin_edges)
    print(f'truncate_non_sec: Sampled {len(idxs)} full-length non-selenoproteins for truncation.')
    # Assign each of the sampled proteins a bin label, where the bins are the same as those of the full-length selenoprotein
    # length distribution (from above).
    df = df.iloc[idxs].copy()
    df['bin_label'], _ = digitize(df.seq.apply(len).values, bin_edges)

    # Generate a continuous distribution for truncation ratios for each length bin. This is necessary because of 
    # heteroscedacity: the variance in the truncation ratios of short sequences is much higher than that of long
    # sequences, so need to generate different distributions for different length categories. 
    kdes = dict()

    pbar = tqdm(total=len(np.unique(bin_labels)), desc='truncate_non_sec: Generating KDEs of length bins...')
    for bin_label, bin_values in groupby(sec_truncation_ratios, bin_labels).items():
        # print(f'truncate_non_sec: Fitting KDE for length bin {bin_label}.')
        kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth) 
        kde.fit(bin_values.reshape(-1, 1))
        kdes[bin_label] = kde
        pbar.update(1)
    # pbar.close()

    # Use the KDE to sample truncation ratios for each length bin, and apply the truncation to the full-length sequence. 
    df_truncated = []
    pbar = tqdm(total=len(df), desc='truncate_non_sec: Sampling truncation sizes from KDEs...')
    for bin_label, bin_df in df.groupby('bin_label'):
        # print(f'truncate_non_sec: Sampling {len(bin_df)} truncation sizes from KDE.')
        bin_df['truncation_size'] = kdes[bin_label].sample(n_samples=len(bin_df), random_state=42).ravel() * bin_df.seq.apply(len).values
        bin_df['seq'] = bin_df.apply(lambda row : row.seq[:-int(row.truncation_size)], axis=1)
        df_truncated.append(bin_df)
        pbar.update(len(bin_df))
    pbar.close()

    print(f'truncate_non_sec: Creating DataFrame of truncated non-selenoproteins.')
    df_truncated = pd.concat(df_truncated).drop(columns=['bin_label'])
    df_truncated.index.name = 'id'
    df_truncated.index = [id_ + TRUNCATED_SYMBOL for id_ in df_truncated.index]
    # print(f'truncate_non_sec: Complete.')
    return df_truncated


def split(df:pd.DataFrame, data_dir:str=None, overwrite:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Divide the uniprot data into training, testing, and validation datasets. The split is cluster-aware, and ensures that
    no CD-HIT-generated cluster is split between datasets. The data is first divided into a training set (80 percent) and test
    set (20 percent), and the training data is further divided into a training (80 percent) and validation (20 percent) set.'''

    clusterer = Clusterer(tool='mmseqs', name='all_labels', cwd=data_dir)
    df = clusterer.run(df, overwrite=overwrite)
    
    n = len(df) # Get the original number of sequences for checking stuff later. 
    groups = df['mmseqs_cluster'].values # Extract cluster labels. 
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8)

    idxs, test_idxs = list(gss.split(df.values, groups=groups))[0]
    test_df = df.iloc[test_idxs].copy()
    df, groups = df.iloc[idxs].copy(), groups[idxs] # Now working only with the remaining sequence data, not in the test set. 
    assert (len(df) + len(test_df)) == n, f'split: The size of the remaining dataset is incorrect. len(df) + len(test_df) is {len(df) + len(test_df)}, but should equal {n}.'
    
    train_idxs, val_idxs = list(gss.split(df.values, groups=groups))[0]

    train_df = df.iloc[train_idxs].copy()
    val_df = df.iloc[val_idxs].copy() 
    
    assert len(np.intersect1d(train_df.index, val_df.index)) == 0, 'split: There is leakage between the validation and training datasets.'
    assert len(np.intersect1d(train_df.index, test_df.index)) == 0, 'split: There is leakage between the testing and training datasets.'
    assert len(np.intersect1d(test_df.index, val_df.index)) == 0, 'split: There is leakage between the validation and testing datasets.'
    
    return train_df, test_df, val_df


def stats(df:pd.DataFrame, name:str=None):
    print(f'\nstats: Information for {name}')

    size = len(df)
    mean_cluster_size = np.round(df.groupby('mmseqs_cluster').apply(len).mean(), 2)
    n_clusters = len(df.mmseqs_cluster.unique()) 

    print(f'stats: Dataset size:', size)
    print(f'stats: Mean cluster size (80%):', mean_cluster_size)
    print(f'stats: Number of clusters (80%):', n_clusters)
    for label, label_df in df.groupby('label'):
        label_fraction = np.round(len(label_df) / size, 2)
        print(f'stats: Fraction of dataset with label {label}: {label_fraction}')



# NOTE: Should I dereplicate the selenoproteins before or after truncation? Seems like after would be a good idea.
def process(path:str, data_dir:str=None, label:int=None, overwrite:bool=False, **kwargs):

    print(f'process: Processing dataset {path}...')

    df = clean(pd.read_csv(path, index_col=0), **kwargs)

    if label == 1: # Truncate if the dataset is for category 1 or 2. 
        df = truncate_sec(df)
    elif label == 2:
        df = truncate_non_sec(df, **kwargs)

    clusterer = Clusterer(name=f'label_{label}', cwd=data_dir, tool='mmseqs')
    df = clusterer.dereplicate(df, overwrite=overwrite) # Don't cluster by homology just yet. 
    df['label'] = label # Add labels to the data marking the category. 

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print-stats', action='store_true')
    parser.add_argument('--save-metadata', action='store_true')
    parser.add_argument('--embed', action='store_true')
    args = parser.parse_args()

    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')

    label_0_df = process(uniprot_sprot_path, label=0, data_dir=args.data_dir, overwrite=args.overwrite)
    label_1_df = process(uniprot_sec_path, label=1, data_dir=args.data_dir, allow_c_terminal_fragments=True, overwrite=args.overwrite)
    label_2_df = process(uniprot_sprot_path, label=2, data_dir=args.data_dir, allow_c_terminal_fragments=True, remove_selenoproteins=True, label_1_df=label_1_df, overwrite=args.overwrite)
    df = pd.concat([label_0_df, label_1_df, label_2_df])

    train_df, test_df, val_df = split(df, data_dir=args.data_dir, overwrite=args.overwrite)
    datasets = {'train':train_df, 'test':test_df, 'val':val_df}

    if args.print_stats:
        for file_name, df in datasets.items():
            stats(df, name=file_name)

    if args.save_metadata:
        # NOTE: I want to be able to add to exsting HDF files. 
        for file_name, df in datasets.items():
            # path = os.path.join(args.data_dir, file_name + '.csv')
            path = os.path.join('.', file_name + '.csv')
            df.to_csv(path)

    if args.embed:
        # NOTE: I want to be able to add to exsting HDF files. 
        for file_name, df in datasets.items():
            path = os.path.join(args.data_dir, file_name + '.h5')
            embed(df, path=path, append=args.append)
