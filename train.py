#!/usr/bin/env python
'''Small script which loads the training data, trains the input model, pickles the resulting reporter object, and saves the model weights.'''
from selenobot.classifiers import Classifier, SimpleClassifier
from selenobot.dataset import get_dataloader, Dataset
from selenobot.utils import load_config_paths
import os
import pickle
import numpy as np
import torch
import argparse

DETECT_DATA_DIR = load_config_paths()['detect_data_dir']
WEIGHTS_DIR = load_config_paths()['weights_dir']
REPORTERS_DIR = load_config_paths()['reporters_dir']

def train(
    model:torch.nn.Module,
    embedder:str=None,
    epochs:int=None,
    selenoprotein_fraction:float=None,
    weights_path:str=None,
    reporter_path:str=None):

    train_dataloader = get_dataloader(TRAIN_PATH, balance_batches=True, selenoprotein_fraction=selenoprotein_fraction, batch_size=1024, embedder=embedder)
    
    train_reporter = model.fit(train_dataloader, val_dataset=Dataset(pd.read_csv(VAL_PATH), embedder=embedder), epochs=epochs, lr=0.001)
    # Save the reporter and model weights. 
    with open(reporter_path, 'wb') as f:
        pickle.dump(train_reporter, f)
    torch.save(model.state_dict(), model_weights_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['plm', 'aac', 'length'], type=str, help='Specifies the model type to be trainied on the data.')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train the specified model.', default=10)
    args = parser.parse_args()

    # Load the model stuff based on the input. 
    if args.model == 'aac':
        model = Classifier(latent_dim=21, hidden_dim=8)
    elif args.model == 'length':
        model = SimpleClassifier(latent_dim=1)
    elif args.model == 'plm':
        model = Classifier(latent_dim=1024, hidden_dim=512)

    kwargs = {'epochs':args.epochs}
    kwargs['embedder'] = args.model
    kwargs['weights_path'] = os.path.join(WEIGHTS_DIR, args.model + '.pth')
    kwargs['reporters_path'] = os.path.join(REPORTERS_DIR, args.model + '_training.pkl')
    kwargs['selenoprotein_fraction'] = 0.5

    train(model, **kwargs)

