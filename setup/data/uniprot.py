'''Code for setting up the uniprot data subdirectory.'''
import pandas as pd
import numpy as np
import re
import random
import os
from selenobot.utils import fasta_concatenate, pd_to_fasta, pd_from_fasta
from configparser import ConfigParser


def setup_sec_truncated(config:ConfigParser, first_sec_only:bool=True) -> ConfigParser:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all 
    sequences contained in the file contain selenocysteine, labeled as U.'''

    sec_truncated_path = os.path.join(config['paths']['uniprot_data_dir'], 'sec_truncated.fasta')

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
    
    df = pd.DataFrame(df_trunc).set_index('id')
    pd_to_fasta(df, path=sec_truncated_path)

    # Add the new path to the configuration file.
    config['paths']['sec_truncated_path'] = sec_truncated_path

    return config


def setup_sprot(config:ConfigParser) -> ConfigParser:
    '''Preprocessing for the SwissProt sequence data. Removes all selenoproteins from the SwissProt database.'''
    sprot_data = pd_from_fasta(config['paths']['sprot_path'])
    # Remove all sequences containing U (selenoproteins) from the SwissProt file. 
    selenoproteins = sprot_data['seq'].str.contains('U')
    if np.sum(selenoproteins) > 0:
        sprot_data = sprot_data[~selenoproteins]
        print(f'setup.data.uniprot.setup_sprot: {np.sum(selenoproteins)} selenoproteins successfully removed from SwissProt.')
        # Overwrite the original file. 
        pd_to_fasta(sprot_data, path=config['paths']['sprot_path'])


def setup_uniprot(config:ConfigParser) -> ConfigParser:
    '''Set up all data in the uniprot subdirectory, and modify the configuration file to contain the new file paths.'''
    
    # Setup the file of truncated selenoproteins. 
    config = setup_sec_truncated(config)
    config = setup_sprot(sprot_path=os.path.join(UNIPROT_DATA_DIR, 'sprot.fasta'))
    
    all_data_path = os.path.join(config['paths']['uniprot_data_dir'], 'all_data.fasta')
    # Combine all FASTA files required for the project into a single all_data.fasta file. This allows CD-HIT to be applied to everything at once.
    fasta_concatenate([config['paths']['sec_truncated_path'], config['paths']['sprot']], out_path=all_data_path)
    # Add the new file path to the configuration file. 
    config['paths']['all_data_path'] = all_data_path

    return config # Return the modified configuration file.
