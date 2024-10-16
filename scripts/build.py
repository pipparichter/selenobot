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

warnings.simplefilter('ignore')

def clean(sprot_df:pd.DataFrame) -> pd.DataFrame:
    '''Even amongst SwissProt-reviewed proteins, many sequences are incomplete (i.e. have a non-terminal residue feature). 
    These should be removed from the dataset, so that only full-length proteins are considered positive cases.''' 
    n = (~sprot_df.non_terminal_residue.isnull()).sum()
    print(f'clean: Removing {n} fragment proteins from SwissProt. {len(sprot_df) - n} sequences remaining.') 
    sprot_df = sprot_df[sprot_df.non_terminal_residue.isnull()]
    return sprot_df


def truncate(sec_df:pd.DataFrame) -> pd.DataFrame:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all 
    sequences contained in the file contain selenocysteine, labeled as U.'''
    # Want to remove all selenoproteins which are incomplete at the C-terminus; those 
    # which are incomplete at the N-terminus are admissable because we are truncating anyway. 
    n = 0 

    sec_df_truncated = []
    for row in tqdm(sec_df.to_dict(orient='records'), 'truncate: Truncating selenoproteins...'):

        non_terminal_residue_positions = row['non_terminal_residue'].split(',') if (type(row['non_terminal_residue']) == str) else []
        if '1' in non_terminal_residue_positions:
            n += 1
            continue

        row['id'] = row['id'] + '[1]' # Modify the row ID to contain a [1] label, indicating truncation at the first selenocysteine. 
        row['sec_index'] = row['seq'].index('U') # This will raise an exception if no U residue is found.
        row['sec_count'] = row['seq'].count('U') # Store the number of selenoproteins in the original sequence.
        row['trunc'] = len(row['seq']) - row['sec_index'] # Store the number of amino acid residues discarded.
        row['seq'] = row['seq'][:row['sec_index']] # Get the portion of the sequence prior to the U residue.
        sec_df_truncated.append(row)
    
    print(f'clean: Removing {n} fragment proteins from the selenoprotein dataset. {len(sec_df_truncated) - n} sequences remaining.')    
    return pd.DataFrame(sec_df_truncated)



def cluster(path:str, n:int=5, c:float=0.8, l:int=5) -> NoReturn:
    '''Run the CD-HIT clustering tool on the data stored in the uniprot.fasta file.'''
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    # Run the CD-HIT command with the specified cluster parameters. 
    # CD-HIT can be installed using conda. conda install bioconda::cd-hit
    subprocess.run(f'cd-hit -i {path} -o {os.path.join(directory, name)} -n {n} -c {c} -l {l}', shell=True, check=True, stdout=subprocess.DEVNULL)

    return os.path.join(directory, name + '.clstr')


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    # parser.add_argument('--bacteria-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    bacteria_only = True

    if (not os.path.exists(os.path.join(args.data_dir, 'uniprot.fa'))) or args.overwrite:
        # There are duplicates here, as there were multiple accessions for the same protein. 
        sec_df = pd.read_csv(os.path.join(args.data_dir, 'uniprot_sec.csv')).drop_duplicates('name', keep='first')
        print(f'Loaded data from {os.path.join(args.data_dir, 'uniprot_sec.csv')}')
        sprot_df = pd.read_csv(os.path.join(args.data_dir, 'uniprot_sprot.csv')).drop_duplicates('name', keep='first')
        print(f'Loaded data from {os.path.join(args.data_dir, 'uniprot_sprot.csv')}')
        sprot_df = clean(sprot_df) # Make sure all SwissProt proteins are complete (i.e. not fragments).

        if bacteria_only:
            print('Filtering out all sequences which do not belong to bacteria.')
            sec_df = sec_df[sec_df.domain == 'Bacteria']
            sprot_df = sprot_df[sprot_df.domain == 'Bacteria']

        selenoproteins_in_sprot = np.isin(sprot_df['id'].values, sec_df['id'].values)
        print('Removing', np.sum(selenoproteins_in_sprot), 'selenoproteins from the SwissProt data.')
        sprot_df = sprot_df[~selenoproteins_in_sprot]

        sec_df = truncate(sec_df) # Truncate the selenoproteins at the first selenocysteine residue. 
        df = pd.concat([sec_df, sprot_df], axis=0).set_index('id') # Combine all the data into a single DataFrame. 
        print(f'Merged selenoprotein and SwissProt DataFrames, {len(df)} total sequences.')

        fasta_file = FastaFile.from_df(df)
        fasta_file.write(os.path.join(args.data_dir, 'uniprot.fa'))
    else:
        df = FastaFile(os.path.join(args.data_dir, 'uniprot.fa')).to_df()

    if (not os.path.exists(os.path.join(args.data_dir, 'uniprot.clstr'))) or args.overwrite:
        cluster(os.path.join(args.data_dir, 'uniprot.fa'))
    clstr_df = CdhitClstrFile(os.path.join(args.data_dir, 'uniprot.clstr')).to_df()

    df = df.merge(clstr_df, left_index=True, right_index=True, how='inner')
    df['label'] = np.array(['[1]' in id_ for id_ in df.index]).astype(int) # Add a label column, indicating truncated or not. 
    print(f'Added cluster labels to the DataFrame, {len(df)} total sequences.')

    train_df, test_df, val_df = split(df)

    for filename, df in zip(['train.h5', 'test.h5', 'val.h5'], [train_df, test_df, val_df]):
        embed(df, path=os.path.join(args.data_dir, filename))
        print(f'Dataset saved to {os.path.join(args.data_dir, filename)}')
