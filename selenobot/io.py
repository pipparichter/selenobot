'''Utilities for accessing data stored in the Google Cloud bucket, and setting up the configuration file with the corresponding file paths.'''
import subprocess
import configparser
import os
import pathlib

# This assumes that this script is being run from the project root directory, so this should be an absolute path to selenobot.
PROJECT_DIR = os.getcwd()
CONFIG_FILE_PATH = os.path.join(PROJECT_DIR, 'selenobot.cfg')

def download_training_data(data_directory):
    pass


def download_testing_data(data_directory):
    pass


def download_gtdb_data(data_directory):
    pass


def configure(data_dir=None):
    '''Set up the configuration file.'''
    config = configparser.ConfigParser()

    if os.path.exists(cfg_file_path):
        config.read(cfg_file_path)

    config['paths'].update({'data_dir':data_dir})
    config['paths'].update({'detect_data_dir':os.path.join(data_dir, 'detect')})
    config['paths'].update({'weights_dir':os.path.join(PROJECT_DIR, 'weights')})
    config['paths'].update({'reporters_dir':os.path.join(PROJECT_DIR, 'reporters')})

    # Make all necessary directories.
    for dir_name, path in config['paths'].items():
        if not os.path.isdir(path) and 'dir' in dir_name:
            os.mkdirs(path)

    # weights_dir = /home/prichter/Documents/selenobot/weights/
    # reporters_dir = /home/prichter/Documents/selenobot/reporters/

    with open(cfg_file_path, 'w') as f:
        config.write(f)

if __name__ == '__main__':
    print(os.getcwd())
