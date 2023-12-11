'''Contains all the functions that a selenobot user would need to apply the model or reproduce the results. This prevents the user from having to import 
multiple disparate models, and makes it easier to centralize error handling.'''
import pandas as pd
import os
import numpy as np
import random
import scipy.stats
from typing import Union

from selenobot.utils import *
import selenobot.embedders as embedders
import selenobot.dataset
import selenobot.classifiers as classifiers
import selenobot.reporters as reporters


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


def create_embedder(type_:str=None) -> embedders.Embedder:
    '''Instantiate an Embedder of the specified type.
    
    :param type_: The type of embedder to create. Can be 'aac', 'len', or None. 
    :return: An Embedder object of the specified type or None. 
    :raises ValueError: Throws an error if the specified type_ is invalid. 
    '''
    if type_ is None: # This allows for consistency when working with PLM embeddings. 
        return None
    if type_ == 'aac':
        return embedders.AacEmbedder()
    if type_ == 'len':
        return embedders.LengthEmbedder()
    else:
        raise ValueError(f"selenobot.create_embedder: Input type must be one of 'aac' or 'len' (or left as None).")


def create_dataset(embedder:Union[embedders.Embedder, None], path:str=None, nrows:int=None) -> selenobot.dataset.Dataset:
    '''Instantiate a Dataset using the specified embedder and the data at the specified path.
    
    :param embedder: An Embedder object specifying the embedding method to apply to the data. 
    :param path: The path to the data (a CSV file) to load into the Dataset object.
    :param nrows: The number of rows to load in from the CSV file at path. If not specified, all rows are loaded. 
    :return: A Dataset object with the specified number of rows and embedding type. 
    '''

    # Collect all the information into a single pandas DataFrame. Avoid loading all embeddings if not needed.
    if nrows is not None:
        # Should make sure rows are sampled randomly to increase likelihood that there are some truncated and some full-length. 
        n = csv_size(path)
        rows = set(random.sample(list(range(n)), nrows)) # Without replacement by default. 
        skiprows = set(range(n)) - rows - set([0]) # Hopefully this does not skip the header?
        data = pd.read_csv(path, usecols=None if embedder is None else ['seq', 'label', 'id'], skiprows=skiprows)
    else:
        data = pd.read_csv(path, usecols=None if embedder is None else ['seq', 'label', 'id'])

    dataset = selenobot.dataset.Dataset(data, embedder=embedder)
    return dataset # Can only return a Dataset with a properly-specified type, which makes the user less likely to break the pipeline.


def create_classifier(dataset:selenobot.dataset.Dataset) -> classifiers.Classifier:
    '''Instantiate a Classifier capable of classifying the data in the input dataset.
    
    :param dataset: A Dataset object, which is used to determine the layer dimensions.
    :return: A Classifier object with the appropriate layer dimensions.
    '''
    # Using None as default for PLM classifiers might help with user interface consistency.
    if dataset.type == 'plm':
        return classifiers.Classifier(hidden_dim=512, latent_dim=1024)
    elif dataset.type == 'aac':
        return classifiers.Classifier(latent_dim=21, hidden_dim=8)
    elif dataset.type == 'len':
        return classifiers.SimpleClassifier(latent_dim=1)


def train(
    model:classifiers.Classifier, # Will be the output of create_classifier.
    dataset:selenobot.dataset.Dataset, # Will be the output of create_dataset. 
    val_dataset:selenobot.dataset.Dataset=None,
    epochs:int=10,
    batch_size:int=1024,
    balance_batches=True) -> reporters.TrainReporter:
    '''Train the input model with the specified parameters.
    
    :param model: A Classifier to train on the input Datasets. 
    :param dataset: A Dataset containing the training data. 
    :param val_dataset: A Dataset containing the validation data. 
    :param epochs: The number of epochs to train the model for. 
    :param batch_size: The size of the batches which the training data will be split into. 
    :param balance_batches: Whether or not to ensure that each batch has equal proportion of full-length and truncated proteins. 
    :return: A TrainReporter object containing information about the training procedure. 
    :raises AssertionError: If the types of the validation and training Datasets don't match. 
    '''

    assert val_dataset.type == dataset.type, 'selenobot.train: Types of the validation and training datasets must match.'

    dataloader = selenobot.dataset.get_dataloader(dataset, balance_batches=balance_batches, selenoprotein_fraction=0.5, batch_size=batch_size)
    reporter = model.fit(dataloader, val_dataset=val_dataset, epochs=epochs, lr=0.001)
    return reporter # Return the reporter object with the training information. 


def get_train_data_path() -> str:
    '''Return the path to the training data.'''
    return load_config_paths()['train_path']


def get_test_data_path() -> str:
    '''Return the path to the testing data.'''
    return load_config_paths()['test_path']


def get_val_data_path() -> str:
    '''Return the path to the validation data.'''
    return load_config_paths()['val_path']

# if __name__ == '__main__':
#     get_test_data_summary()
#     get_train_data_summary()
#     get_val_data_summary()