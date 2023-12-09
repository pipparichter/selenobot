'''Script for setting up the data directory system, and loading relevant directory information into the configuration file. '''
import subprocess
from configparser import ConfigParser
import os
import typing
from time import perf_counter
import requests
import numpy as np
import argparse
import zipfile
from tqdm import tqdm

import sys
sys.path.append('./data/')
import detect, uniprot

# This assumes that this script is being run from the project root directory, so this should be an absolute path to selenobot.
PROJECT_DIR = os.getcwd()
CONFIG_FILE_PATH = os.path.join(PROJECT_DIR, 'selenobot.cfg')

# UNIPROT VERSION 2023 3
# GTDB VERSION r207

BUCKET = 'https://storage.googleapis.com/selenobot-data/'

# def unzip_file(path:str, format='csv') -> str:
#     '''Unzip a file at the specified path. Returns the path to the extracted file.'''
#     directory = os.path.dirname(path)
#     # Uses the zip file name to establish a new file name. 
#     filename = os.path.basename(path).replace('.zip', f'.{format}')
#     with zipfile.ZipFile(path, 'r') as f:
#         # Extract the file to the same directory. 
#         f.extractall(path=os.path.join(directory, filename))
    
#     return os.path.join(directory, filename) # Returns the path to the extracted file.


# def write_zip_file(path:str, response:requests.Response) -> str:
#     '''Write a compressed file from the Google Cloud Bucket to the specified path.'''
#     chunk_size = 1024 * 1024 # About 1 MB, somewhat arbitrary. 
#     content_length = int(response.headers.get('Content-Length')) # Total number of bytes. 

#     with open(path, 'wb') as f: # Write the zip file. 
#         for chunk in tqdm(response.iter_content(chunk_size=1024 * 1024), desc='setup.main.write_zip_file', leave=True, total=content_length//chunk_size):
#             f.write(chunk)
#         # Make sure the entire streamed content was downloaded was downloaded.
#         assert f.tell() == content_length, 'setup.main.write_zip_file: Incomplete download.'

#     path = unzip_file(path) # Unzip the new file, and get the path of the unzipped file. 
#     print(f'setup.main.write_zip_file: Extracted file to {path}.')
#     return path


def write_file(path:str, response:requests.Response, stream:bool=False) -> str:
    '''Write a non-compressed file from the Google Cloud Bucket to the specified path.'''
    if not stream:
        with open(path, 'w') as f:
            f.write(response.text)
        
    else:
        chunk_size = 1024 * 1024 # About 1 MB, somewhat arbitrary. 
        content_length = int(response.headers.get('Content-Length')) # Total number of bytes. 

        with open(path, 'wb') as f: # Write the zip file. 
            for chunk in tqdm(response.iter_content(chunk_size=1024 * 1024), desc='setup.main.write_file', leave=True, total=content_length//chunk_size):
                f.write(chunk)
            # Make sure the entire streamed content was downloaded was downloaded.
            assert f.tell() == content_length, 'setup.main.write_file: Incomplete download.'

    return path



def download(filename:str, directory:str, config:ConfigParser, stream:bool=False) -> ConfigParser:
    '''Download a file from the Google Cloud bucket, writing the information to the specified directory.
    Also adds the resulting filepath to the config file.
    
    args:
        - filename: The name of the file in the Google Cloud bucket.
        - directory: The directory location where the downloaded file will be stored. 
        - config: The configuration file object.
        - stream: Whether or not to stream the downloaded file to avoid excessive RAM usage. 
    '''
    path = os.path.join(directory, filename)
    if not os.path.exists(path): # Skip the download step if file is already present. 
        t1 = perf_counter()
        response = requests.get(BUCKET + filename, stream=stream)
        path = write_file(path, response, stream=stream)
        t2 = perf_counter()
        print(f'setup.main.download: Downloaded {filename} to {directory} in {np.round(t2 - t1, 2)} seconds.')
        
    key = filename.split('.')[0] + '_path' # Create a key for the path in the config file by removing the file extension.
    config['paths'][key] = path # This should be the extracted file's path, when unzip=True.
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


def setup(data_dir, cdhit=None):
    '''Set up the data directories and configuration file.'''

    config = ConfigParser() # Start the collector for the configuration information. 
    config['paths'] = {}
    config['cdhit'] = {'cdhit_min_seq_length':'6', 'cdhit_sequence_similarity':'0.8', 'cdhit_word_length':'5', 'cdhit':cdhit}

    config = setup_data_directory(data_dir, config=config)
    uniprot_data_dir = config['paths']['uniprot_data_dir']
    gtdb_data_dir = config['paths']['gtdb_data_dir']

    # Download all necessary files to the uniprot subdirectory. 
    config = download('embeddings.zip', uniprot_data_dir, config, stream=True)
    config = download('embeddings.csv', uniprot_data_dir, config, stream=True)
    config = download('sec.fasta', uniprot_data_dir, config)
    config = download('sprot.fasta', uniprot_data_dir, config)
    # Download all necessary files to the gtdb subdirectory. 
    config = download('archaea_tree.txt', gtdb_data_dir, config)
    config = download('bacteria_tree.txt', gtdb_data_dir, config)
    config = download('archaea_metadata.tsv', gtdb_data_dir, config)
    config = download('bacteria_metadata.tsv', gtdb_data_dir, config)

    config = uniprot.setup(config)
    config = detect.setup(config)

    # Write the configuration file. 
    with open(CONFIG_FILE_PATH, 'w') as f:
        config.write(f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Specifies the path of the data directory.')
    parser.add_argument('--cdhit', type=str, help='Path to the cdhit command.', default='~/cd-hit-v4.8.1-2019-0228/cd-hit')
    args = parser.parse_args()
    
    setup(args.data_dir, cdhit=args.cdhit)

