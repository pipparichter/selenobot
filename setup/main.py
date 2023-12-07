'''Script for setting up the data directory system, and loading relevant directory information into the configuration file. '''
import subprocess
from configparser import ConfigParser
import os
import typing
from time import perf_counter
import requests
import numpy as np
import argparse

from selenobot.setup.data.detect import setup_detect
from selenobot.setup.data.uniprot import setup_uniprot

# This assumes that this script is being run from the project root directory, so this should be an absolute path to selenobot.
PROJECT_DIR = os.getcwd()
CONFIG_FILE_PATH = os.path.join(PROJECT_DIR, 'selenobot.cfg')

# UNIPROT VERSION 2023 3
# GTDB VERSION r207

BUCKET = 'https://storage.googleapis.com/selenobot-data/'

def download_file(filename:str, directory:str, config:ConfigParser) -> ConfigParser:
    '''Download a file from the Google Cloud bucket, writing the information to the specified directory.
    Also adds the resulting filepath to the config file'''

    print(f'setup.data.download_file: Downloading {filename} from Google Cloud.')
    url = BUCKET + filename
    path = os.path.join(directory, filename)

    if os.path.exists(path): # Don't try to overwrite. 
        print(f'setup.data.download_file: {path} already exists.')
    else:
        t1 = perf_counter()
        response = requests.get(url)
        with open(path, 'w') as f:
            f.write(response.text)
        t2 = perf_counter()
        print(f'setup.data.download_file: Downloaded {filename} to {directory} in {np.round(t2 - t1, 2)} seconds.')

    # Create a key for the path in the config file by removing the file extension.
    key = filename.split('.')[0] + '_path'
    config['paths'][key] = path
    return config


def setup_data_directory(data_dir:str, config:ConfigParser=None) -> ConfigParser:
    '''Set up the data directory structure, starting from the specified root directory.'''

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir) # Make the data directory (and any necessary intermediates).

    config['paths']['data_dir'] = data_dir
    config['paths']['detect_data_dir'] = os.path.join(data_dir, 'detect')
    config['paths']['uniprot_data_dir'] = os.path.join(data_dir, 'uniprot')
    config['paths']['gtdb_data_dir'] = os.path.join(data_dir, 'gtdb')

    # Make all remaining subdirectories.
    for dir_path in config['paths'].values():
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    return config


def setup(data_dir, cdhit='/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'):
    '''Set up the data directories and configuration file.'''

    config = ConfigParser() # Start the collector for the configuration information. 
    config['paths'] = {}
    config['cdhit'] = {'cdhit_min_seq_length':'6', 'cdhit_sequence_similarity':'0.8', 'cdhit_word_length':'5', 'cdhit':cdhit}

    config = setup_data_directory(data_dir, config=config)
    uniprot_data_dir = config['paths']['uniprot_data_dir']
    gtdb_data_dir = config['paths']['gtdb_data_dir']

    # Download all necessary files to the uniprot subdirectory. 
    config = download_file('embeddings.csv', uniprot_data_dir, config)
    config = download_file('sec.fasta', uniprot_data_dir, config)
    config = download_file('sprot.fasta', uniprot_data_dir, config)
    # Download all necessary files to the gtdb subdirectory. 
    config = download_file('archaea_tree.txt', gtdb_data_dir, config)
    config = download_file('bacteria_tree.txt', gtdb_data_dir, config)
    config = download_file('archaea_metadata.tsv', gtdb_data_dir, config)
    config = download_file('bacteria_metadata.tsv', gtdb_data_dir, config)

    config = setup_uniprot(config)
    config = setup_detect(config)

    # Write the configuration file. 
    with open(CONFIG_FILE_PATH, 'w') as f:
        config.write(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Specifies the path of the data directory.')
    parser.add_argument('--cdhit', type=str, help='Path to the cdhit command.', default='~')
    args = parser.parse_args()
    
    setup(args.data_dir, cdhit=args.cdhit)

