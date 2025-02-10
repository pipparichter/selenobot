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
import glob
import warnings
from selenobot.tools import Clusterer
from selenobot.utils import seed, truncate_sec


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


def is_selenoprotein(seq:str):
    return 'U' in seq

def is_short(seq:str):
    return len(seq) < 200


# NOTE: Did I initially have a reason for de-replicating everything separately?
# NOTE: Should I de-replicate before or after truncation? I don't think it should matter much...

def dereplicate(metadata_df:pd.DataFrame, overwrite:bool=False):

    post_derep_path = os.path.join(DATA_DIR, f'{MODE}_metadata.derep.csv')
    
    if (not os.path.exists(post_derep_path)) or overwrite:
        clusterer = Clusterer(name=f'{MODE}', cwd=DATA_DIR)
        metadata_derep_df = clusterer.dereplicate(metadata_df, sequence_identity=0.95) 
        clusterer.cleanup() # Remove extraneous output files.  
        metadata_derep_df.to_csv(post_derep_path)
    else:
        metadata_derep_df = pd.read_csv(post_derep_path, index_col=0)

    n_pre_derep, n_post_derep = len(metadata_df), len(metadata_derep_df)
    print(f'dereplicate: Dereplication at 0.95 sequence identity eliminated {n_pre_derep - n_post_derep} sequences. {n_post_derep} sequences remaining.')
    return metadata_derep_df


def clean(metadata_df:pd.DataFrame, min_length:int=MIN_LENGTH, max_length:int=MAX_LENGTH, overwrite:bool=False) -> pd.DataFrame:
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

    metadata_df = dereplicate(metadata_df, overwrite=overwrite)

    return metadata_df


def load(paths:list, n_classes:int=2, overwrite:bool=False):

    metadata_df = list()
    for path, path_parsed in zip(paths, [path.replace('.xml', '.csv') for path in paths]):
        if not os.path.exists(path_parsed):
            df = XMLFile(path).to_df() 
            df.to_csv(path_parsed)
        else:
            df = pd.read_csv(path_parsed, index_col=0)
        metadata_df.append(df)
    metadata_df = pd.concat(metadata_df)
    metadata_df = clean(metadata_df, overwrite=overwrite)

    metadata = dict()
    if n_classes == 2:
        metadata[0] = metadata_df[~metadata_df.seq.apply(is_selenoprotein)].copy()
        metadata[1] = metadata_df[metadata_df.seq.apply(is_selenoprotein)].copy()
    if n_classes == 3:
        metadata[0] = metadata_df[~metadata_df.seq.apply(is_selenoprotein) & ~metadata_df.seq.apply(is_short)].copy()
        metadata[1] = metadata_df[metadata_df.seq.apply(is_selenoprotein)].copy()
        metadata[2] = metadata_df[~metadata_df.seq.apply(is_selenoprotein) & metadata_df.seq.apply(is_short)].copy()
    if n_classes == 4:
        metadata[0] = metadata_df[~metadata_df.seq.apply(is_selenoprotein) & ~metadata_df.seq.apply(is_short)].copy()
        metadata[1] = metadata_df[metadata_df.seq.apply(is_selenoprotein)].copy()
        metadata[2] = metadata_df[~metadata_df.seq.apply(is_selenoprotein) & metadata_df.seq.apply(is_short)].copy()
        metadata[3] = metadata_df[metadata_df.seq.apply(is_selenoprotein)].copy()
    
    metadata = {label:df.assign(label=label) for label, df in metadata.items()}
    return metadata


def split(metadata_df:pd.DataFrame, overwrite:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Divide the uniprot data into training, testing, and validation datasets.'''

    cluster_path = os.path.join(DATA_DIR, f'{MODE}_metadata.cluster.csv')
    if (not os.path.exists(cluster_path)) or overwrite:
        clusterer = Clusterer(name=f'{MODE}', cwd=DATA_DIR)
        metadata_df = clusterer.cluster(metadata_df)
        metadata_df[['mmseqs_cluster', 'mmseqs_representative']].to_csv(cluster_path)
        clusterer.cleanup()
    else:
        cluster_df = pd.read_csv(cluster_path, index_col=0)
        metadata_df = metadata_df.merge(cluster_df, left_index=True, right_index=True)

    
    n = len(metadata_df) # Get the original number of sequences for checking stuff later. 
    groups = metadata_df['mmseqs_cluster'].values # Extract cluster labels. 
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8)

    idxs, test_idxs = list(gss.split(metadata_df.values, groups=groups))[0]
    test_metadata_df = metadata_df.iloc[test_idxs].copy()
    metadata_df, groups = metadata_df.iloc[idxs].copy(), groups[idxs] # Now working only with the remaining sequence data, not in the test set. 
    
    train_idxs, val_idxs = list(gss.split(metadata_df.values, groups=groups))[0]

    train_metadata_df = metadata_df.iloc[train_idxs].copy()
    val_metadata_df = metadata_df.iloc[val_idxs].copy() 

    return {'train':train_metadata_df, 'test':test_metadata_df, 'val':val_metadata_df}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n-classes', default=2, type=int)
    parser.add_argument('--add-trembl-short', action='store_true')
    args = parser.parse_args()

    global MODE 
    MODE = f'{args.n_classes}c'
    MODE = MODE + '_xl' if args.add_trembl_short else MODE

    global DATA_DIR
    DATA_DIR = args.data_dir

    file_names = ['uniprot_sprot.xml', 'uniprot_sec.xml'] 
    if args.add_trembl_short:
        file_names += [os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, 'uniprot_trembl_short*'))] 
    paths = [os.path.join(DATA_DIR, file_name) for file_name in file_names]
    
    metadata = load(paths, n_classes=args.n_classes, overwrite=args.overwrite)
    if 3 in metadata:
        metadata[3] = truncate_sec(metadata[3], terminus='n', min_length=MIN_LENGTH)
    if 1 in metadata:
        metadata[1] = truncate_sec(metadata[1], terminus='c', min_length=MIN_LENGTH)
    metadata_df = pd.concat(list(metadata.values()))
   
    datasets = split(metadata_df, overwrite=args.overwrite)
    datasets = {dataset:df.assign(dataset=dataset) for dataset, df in datasets.items()}

    # Write everything to a file, in addition to saving invididually. 
    pd.concat(list(datasets.values())).to_csv(os.path.join(DATA_DIR, f'{MODE}_metadata.csv'))
    for dataset, df in datasets.items():
        df.to_csv(os.path.join(DATA_DIR, f'{MODE}_metadata_{dataset}.csv'))

