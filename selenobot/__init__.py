'''Contains all the functions that a selenobot user would need to apply the model or reproduce the results. This prevents the user from having to import 
multiple disparate models, and makes it easier to centralize error handling.'''
import pandas as pd
import os
import numpy as np
import scipy.stats
from typing import Union

from selenobot.utils import *
import selenobot.embedders as embedders
import selenobot.dataset as dataset
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
    '''Instantiate an Embedder of the specified type.'''
    if type_ is None: # This allows for consistency when working with PLM embeddings. 
        return None
    if type_ == 'aac':
        return embedders.AacEmbedder()
    if type__ == 'len':
        return embedders.LengthEmbedder()
    else:
        raise ValueError(f"selenobot.create_embedder: Input type must be one of 'aac' or 'len' (or left as None).")


def create_dataset(embedder:Union[embedder.Embedder, None], path:str=None) -> dataset.Dataset:
    '''Instantiate a Dataset using the specified embedder and the data at the specified path.'''

    # Collect all the information into a single pandas DataFrame. Avoid loading all embeddings if not needed.
    data = pd.read_csv(path, usecols=None if embedder is None else ['seq', 'label', 'id'])
    dataset = Dataset(data, embedder=embedder)
    return dataset # Can only return a Dataset with a properly-specified type, which makes the user less likely to break the pipeline.


def create_classifier(dataset:dataset.Dataset) -> classifiers.Classifier:
    '''Instantiate a Classifier capable of classifying the data in the input dataset.'''
    # Using None as default for PLM classifiers might help with user interface consistency.
    if dataset.type == 'plm':
        return classifiers.Classifier(hidden_dim=512, latent_dim=1024)
    elif dataset.type == 'aac':
        return classifiers.Classifier(latent_dim=21, hidden_dim=8)
    elif dataset.type == 'len':
        return classifiers.SimpleClassifier(latent_dim=1)


def train(
    model:classifiers.Classifier, # Will be the output of create_classifier.
    dataset:dataset.Dataset, # Will be the output of create_dataset. 
    val_dataset:dataset.Dataset=None,
    epochs:int=10,
    batch_size:int=1024,
    balance_batches=True) -> reporters.TrainReporter:
    '''Train the input model with the specified parameters.'''

    assert val_dataset.type == dataset.type, 'selenobot.train: Types of the validation and training datasets must match.'

    dataloader = dataset.get_dataloader(dataset, balance_batches=balance_batches, selenoprotein_fraction=0.5, batch_size=batch_size)
    reporter = model.fit(train_dataloader, val_dataset=val_dataset, epochs=epochs, lr=0.001)
    return reporter # Return the reporter object with the training information. 


def get_train_data_path() -> str:
    '''Return the path to the training data.'''
    return load_config_paths['train_data_path']


def get_test_data_path() -> str:
    '''Return the path to the testing data.'''
    return load_config_paths['test_data_path']


def get_val_data_path() -> str:
    '''Return the path to the validation data.'''
    return load_config_paths['val_data_path']

# if __name__ == '__main__':
#     get_test_data_summary()
#     get_train_data_summary()
#     get_val_data_summary()