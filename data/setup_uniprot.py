'''Code for setting up the detect subdirectory.'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import random
import subprocess
import os
from typing import NoReturn, Dict, List
import sys
import time
# from transformers import T5EncoderModel, T5Tokenizer
from selenobot.utils import *

# Load information from the configuration file. 
UNIPROT_DATA_DIR = load_config_paths()['uniprot_data_dir']
DETECT_DATA_DIR = load_config_paths()['detect_data_dir']

def setup_sec_truncated(
    sec_path:str=None, 
    sec_truncated_path:str=None, 
    first_sec_only:bool=True) -> NoReturn:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all sequences contained
    in the file contain selenocysteine, labeled as U.
    
    args:
        - sec_path: A path to the FASTA file containing all of the selenoproteins.
        - sec_truncated_path: The path to write the new FASTA file to, containing the truncated proteins. 
        - first_sec_only: Whether or not to truncate at the first selenocystein residue only. If false, truncation is sequential.
    '''
    # Load the selenoproteins into a pandas DataFrame. 
    df = pd_from_fasta(sec_path, set_index=False)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():

        seq = row.seq.split('U')

        if first_sec_only:
            df_trunc['id'].append(row.id + '[1]')
            df_trunc['seq'].append(seq[0])
        else:
            # Number of U's in sequence should be one fewer than the length of the split sequence. 
            df_trunc['id'] += [row.id + f'[{i + 1}]' for i in range(len(seq) - 1)]
            df_trunc['seq'] += ['U'.join(seq[i:i + 1]) for i in range(len(seq) - 1)]
    
    # Do we want the final DataFrame to also contain un-truncated sequences?
    # df = pd.concat([df, pd.DataFrame(df_trunc)]).set_index('id')
    df = pd.DataFrame(df_trunc).set_index('id')
    pd_to_fasta(df, path=sec_truncated_path)


def setup_sprot(sprot_path:str=None) -> NoReturn:
    '''Preprocessing for the SwissProt sequence data. Removes all selenoproteins from the SwissProt database.

    args:
        - sprot_path: Path to the SwissProt FASTA file.
    '''
    f = 'setup.set_sprot'
    sprot_data = pd_from_fasta(sprot_path)
    # Remove all sequences containing U (selenoproteins) from the SwissProt file. 
    selenoproteins = sprot_data['seq'].str.contains('U')
    if np.sum(selenoproteins) == 0:
        print(f'{f}: No selenoproteins found in SwissProt.')
    else:
        print(f'{f}: {np.sum(selenoproteins)} detected out of {len(sprot_data)} total sequences in SwissProt.')
        sprot_data = sprot_data[~selenoproteins]
        assert len(sprot_data) + np.sum(selenoproteins) == fasta_size(sprot_path), f'{f}: Selenoprotein removal unsuccessful.'
        print(f'{f}: Selenoproteins successfully removed from SwissProt.')
        # Overwrite the original file. 
        pd_to_fasta(sprot_data, path=sprot_path)


def setup_uniprot():
    '''Set up all data in the uniprot subdirectory.'''
    
    # Setup the file of truncated selenoproteins. 
    setup_sec_truncated(sec_path=os.path.join(UNIPROT_DATA_DIR, 'sec.fasta'), sec_truncated_path=os.path.join(UNIPROT_DATA_DIR, 'sec_truncated.fasta'))
    setup_sprot(sprot_path=os.path.join(UNIPROT_DATA_DIR, 'sprot.fasta'))
    
    # Combine all FASTA files required for the project into a single all_data.fasta file. 
    fasta_concatenate([os.path.join(UNIPROT_DATA_DIR, 'sec_truncated.fasta'), os.path.join(UNIPROT_DATA_DIR, 'sprot.fasta')], out_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))
    # setup_plm_embeddings(fasta_file_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'), embeddings_path=os.path.join(UNIPROT_DATA_DIR, 'all_embeddings.csv'))

