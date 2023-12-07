import pandas as pd
import os
import numpy as np
import scipy.stats
from selenobot.utils import *
from selenobot.classifiers import Classifier, SimpleClassifier

# Need to do this to load the pickle files I saved under a different directory configuration. 
import sys
from selenobot import reporter
sys.modules['reporter'] = reporter

# TODO: What other information about the databases would be useful to display? 
# Anything about sequence lengths?

def get_train_data_summary():
    '''Print information about the training data.'''
    print('TRAINING DATA SUMMARY')
    detect_data_dir = load_config_paths()['detect_data_dir'] 
    get_data_summary(os.path.join(detect_data_dir, 'train.csv'))


def get_val_data_summary():
    print('VALIDATION DATA SUMMARY')
    '''Print information about the training data.'''
    detect_data_dir = load_config_paths()['detect_data_dir'] 
    get_data_summary(os.path.join(detect_data_dir, 'val.csv'))


def get_test_data_summary():
    print('TESTING DATA SUMMARY')
    '''Print information about the training data.'''
    detect_data_dir = load_config_paths()['detect_data_dir'] 
    get_data_summary(os.path.join(detect_data_dir, 'test.csv'))


def get_data_summary(path:str):
    '''Print information about the data file stored at the given path.'''
    n_selenoproteins = len([id_ for id_ in csv_ids(path) if '[1]' in id_])
    n_total = csv_size(path)

    print('size:', csv_size(path))
    print(f'selenoproteins: {n_selenoproteins} ({int(100 * (n_selenoproteins/n_total))}%)', end='\n\n')


def load_aac_model_weights(path:str) -> Classifier:
    '''Load the amino acid content-based model using pre-generated model weights.'''
    model = Classifier(latent_dim=21, hidden_dim=8)
    return load_model(model, pth_file)


def load_plm_model(path:str) -> Classifier:
    '''Load the PLM-based model using pre-generated model weights.'''
    model = Classifier(hidden_dim=512, latent_dim=1024)
    return load_model(model, pth_file)


def load_length_model(path:str) -> SimpleClassifier:
    '''Load the length-based simple classifier using pre-generated model weights.'''
    # Decided to use simple logistic regression for testing the length. 
    model = SimpleClassifier(latent_dim=1)
    return load_model(model, pth_file)


def load_model(model:Classifier, path:str) -> Classifier:
    '''Helper function for loading models to avoid code duplication.'''
    model.load_state_dict(torch.load(os,path.join(path, pth_file)))
    model.eval() # Put into evaluation mode. 
    return model



# if __name__ == '__main__':
#     get_test_data_summary()
#     get_train_data_summary()
#     get_val_data_summary()