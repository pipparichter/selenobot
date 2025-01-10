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
# Label 2: Truncated non-selenoproteins. 

TRUNCATED_SYMBOL = '-'

warnings.simplefilter('ignore')
seed(42)


# NOTE: C terminus is the end terminus. N terminus is where the methionine is. 
def clean(metadata_df:pd.DataFrame, bacteria_only:bool=True, allow_c_terminal_fragments:bool=False, remove_selenoproteins:bool=False, max_length:int=2000, **kwargs) -> pd.DataFrame:
    '''''' 
    # There are duplicates here, as there were multiple accessions for the same protein. 
    metadata_df = metadata_df.drop_duplicates('name', keep='first')

    if bacteria_only:
        non_bacterial = ~(metadata_df.domain == 'Bacteria')
        metadata_df = metadata_df[~non_bacterial]
        print(f'clean: Removed {non_bacterial.sum()} non-bacterial proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    if remove_selenoproteins: # TODO: Add a check to make sure this catches everything... 
        is_a_selenoprotein = metadata_df.seq.str.contains('U')
        metadata_df = metadata_df[~is_a_selenoprotein]
        print(f'clean: Removed {is_a_selenoprotein.sum()} selenoproteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

    if max_length is not None:
        exceeds_max_length = metadata_df.seq.apply(len) > max_length
        metadata_df = metadata_df[~exceeds_max_length]
        print(f'clean: Removed {exceeds_max_length.sum()} proteins which exceed {max_length} amino acids in length from the DataFrame. {len(metadata_df)} sequences remaining.') 

    # Remove fragmented proteins from the DataFrame.
    # This function checks to see if there is a non-terminal residue at the beginning of the protein (i.e. N-terminal). 
    # If there is not, the fragment is C-terminal. 
    is_c_terminal_fragment = lambda pos : ('1' not in pos.split(',')) if (type(pos) == str) else False
    fragmented = ~metadata_df.non_terminal_residue.isnull() # Get a filter for all proteins which have non-terminal residues. 
    if allow_c_terminal_fragments:
        # If specified, require that the non-terminal residue must be on the N-terminus (beginning) for removal. 
        fragmented = np.logical_and(fragmented, ~metadata_df.non_terminal_residue.apply(is_c_terminal_fragment) )
    metadata_df = metadata_df[~fragmented]
    print(f'clean: Removed {fragmented.sum()} fragment proteins from the DataFrame. {len(metadata_df)} sequences remaining.') 

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


def truncate_non_sec(metadata_df:pd.DataFrame, label_1_metadata_df:pd.DataFrame=None, n_bins:int=10, bandwidth:float=0.01) -> pd.DataFrame:
    '''Sub-sample the set of all full-length proteins such that the length distribution matches that of the full-length
    selenoproteins. Then, truncate the sampled sequences so that the truncated length distributions also match.

    :param metadata_df: The DataFrame containing the complete set of SwissProt proteins. 
    :param sec_metadata_df: 
    :param n_bins: The number of bins to use for producing a length distribution of full-length selenoproteins. 
        This is used when initially down-sampling SwissProt. 
    :param bandwidth: The bandwidth to use for the kernel density estimation, which is used for creating 
        distributions for selecting truncation ratios. 
    :return: A pandas DataFrame containing the sub-sampled and randomly truncated SwissProt proteins. 
    '''
    sec_lengths = label_1_metadata_df.original_length.values
    sec_truncation_ratios = label_1_metadata_df.truncation_ratio.values

    # Group the lengths of the full-length selenoproteins into n_bins bins
    print(f'truncate_non_sec: Generating a selenoprotein length distribution with {n_bins} bins.')
    hist, bin_edges = np.histogram(sec_lengths, bins=n_bins)
    bin_labels, bin_names = digitize(sec_lengths, bin_edges)

    # Sample from the DataFrame of full-length SwissProt proteins, ensuring that the length distribution matches
    # that of the full-length selenoproteins. 
    _, idxs = sample(metadata_df.seq.apply(len).values, hist, bin_edges)
    print(f'truncate_non_sec: Sampled {len(idxs)} full-length non-selenoproteins for truncation.')
    # Assign each of the sampled proteins a bin label, where the bins are the same as those of the full-length selenoprotein
    # length distribution (from above).
    metadata_df = metadata_df.iloc[idxs].copy()
    metadata_df['bin_label'], _ = digitize(metadata_df.seq.apply(len).values, bin_edges)

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
    pbar.close()

    # Use the KDE to sample truncation ratios for each length bin, and apply the truncation to the full-length sequence. 
    metadata_df_truncated = []
    pbar = tqdm(total=len(metadata_df), desc='truncate_non_sec: Sampling truncation sizes from KDEs...')
    for bin_label, bin_metadata_df in metadata_df.groupby('bin_label'):
        # print(f'truncate_non_sec: Sampling {len(bin_metadata_df)} truncation sizes from KDE.')
        bin_metadata_df['truncation_size'] = kdes[bin_label].sample(n_samples=len(bin_metadata_df), random_state=42).ravel() * bin_metadata_df.seq.apply(len).values
        bin_metadata_df['original_length'] = bin_metadata_df.seq.apply(len)
        bin_metadata_df['truncation_ratio'] = bin_metadata_df.truncation_size / bin_metadata_df.original_length 
        bin_metadata_df['seq'] = bin_metadata_df.apply(lambda row : row.seq[:-int(row.truncation_size)], axis=1)
        metadata_df_truncated.append(bin_metadata_df)
        pbar.update(len(bin_metadata_df))
    pbar.close()

    print(f'truncate_non_sec: Creating DataFrame of truncated non-selenoproteins.')
    metadata_df_truncated = pd.concat(metadata_df_truncated).drop(columns=['bin_label'])
    metadata_df_truncated.index.name = 'id'
    metadata_df_truncated.index = [id_ + TRUNCATED_SYMBOL for id_ in metadata_df_truncated.index]
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
    elif label == 2:
        metadata_df = truncate_non_sec(metadata_df, label_1_metadata_df=kwargs.get('label_1_metadata_df'))

    clusterer = Clusterer(name=f'label_{label}', cwd=data_dir, tool='mmseqs')
    metadata_df = clusterer.dereplicate(metadata_df, overwrite=overwrite) # Don't cluster by homology just yet. 
    metadata_df['label'] = label # Add labels to the data marking the category. 

    return metadata_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    uniprot_sprot_path = os.path.join(args.data_dir, 'uniprot_sprot.csv')
    uniprot_sec_path = os.path.join(args.data_dir, 'uniprot_sec.csv')

    label_0_kwargs = {'allow_c_terminal_fragments':False, 'remove_selenoproteins':False}
    label_1_kwargs = {'allow_c_terminal_fragments':True, 'remove_selenoproteins':False}
    label_2_kwargs = {'allow_c_terminal_fragments':True, 'remove_selenoproteins':True}

    label_0_metadata_df = process(uniprot_sprot_path, label=0, data_dir=args.data_dir, overwrite=args.overwrite, **label_0_kwargs)
    label_1_metadata_df = process(uniprot_sec_path, label=1, data_dir=args.data_dir, overwrite=args.overwrite, **label_1_kwargs)
    label_2_metadata_df = process(uniprot_sprot_path, label=2, data_dir=args.data_dir, overwrite=args.overwrite, label_1_metadata_df=label_1_metadata_df, **label_2_kwargs)
    metadata_df = pd.concat([label_0_metadata_df, label_1_metadata_df, label_2_metadata_df])

    train_metadata_df, test_metadata_df, val_metadata_df = split(metadata_df, data_dir=args.data_dir, overwrite=args.overwrite)
    
    metadata = {'train_metadata.csv':train_metadata_df, 'test_metadata.csv':test_metadata_df, 'val_metadata.csv':val_metadata_df}
    for file_name, metadata_df in metadata.items():
        path = os.path.join('.', file_name)
        metadata_df.to_csv(path)
        print(f'Metadata written to {path}')