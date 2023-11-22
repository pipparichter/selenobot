import pandas as pd
import os
import numpy as np
import scipy.stats
from selenobot.utils import *

# Need to do this to load the pickle files I saved under a different directory configuration. 
import sys
from selenobot import reporter
sys.modules['reporter'] = reporter

UNIPROT_DATA_DIR = load_config_paths()['uniprot_data_dir']
DETECT_DATA_DIR = load_config_paths()['detect_data_dir']
WEIGHTS_DIR = load_config_paths()['weights_dir']
REPORTERS_DIR = load_config_paths()['reporters_dir']


# TODO: What other information about the databases would be useful to display? 
# Anything about sequence lengths?


def get_train_data_summary():
    '''Print information about the training data.'''
    print('TRAINING DATA SUMMARY')
    get_data_summary(os.path.join(DETECT_DATA_DIR, 'train.csv'))


def get_val_data_summary():
    print('VALIDATION DATA SUMMARY')
    '''Print information about the training data.'''
    get_data_summary(os.path.join(DETECT_DATA_DIR, 'val.csv'))

def get_test_data_summary():
    print('TESTING DATA SUMMARY')
    '''Print information about the training data.'''
    get_data_summary(os.path.join(DETECT_DATA_DIR, 'test.csv'))


def get_data_summary(path:str):
    '''Print information about the data file stored at the given path.'''
    n_selenoproteins = len([id_ for id_ in csv_ids(path) if '[1]' in id_])
    n_total = csv_size(path)

    print('size:', csv_size(path))
    print(f'selenoproteins: {n_selenoproteins} ({int(100 * (n_selenoproteins/n_total))}%)', end='\n\n')


def load_aac_model_weights(pth_file='aac.pth'):
    '''Load the amino acid content-based model using pre-generated model weights.'''
    model = Classifier(latent_dim=21, hidden_dim=8)
    return load_model(model, pth_file)


def load_plm_model(pth_file='plm.pth'):
    '''Load the PLM-based model using pre-generated model weights.'''
    model = Classifier(hidden_dim=512, latent_dim=1024)
    return load_model(model, pth_file)


def load_length_model(pth_file='length.pth'):
    '''Load the length-based simple classifier using pre-generated model weights.'''
    # Decided to use simple logistic regression for testing the length. 
    model = SimpleClassifier(latent_dim=1)
    return load_model(model, pth_file)


def load_model(model, pth_file):
    '''Helper function for loading models to avoid code duplication.'''
    model.load_state_dict(torch.load(os,path.join(MODEL_WEIGHTS_PATH, pth_file)))
    model.eval() # Put into evaluation mode. 
    return model


def load_train_reporter(pkl_file):
    '''Loads a TrainReporter object from a pickle file in the directory specified in the config file.'''
    with open(os.path.join(REPORTERS_DIR, pkl_file), 'rb') as f:
        train_reporter = pickle.load(f)
    return train_reporter



# if __name__ == '__main__':
#     get_test_data_summary()
#     get_train_data_summary()
#     get_val_data_summary()