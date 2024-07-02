'''A script which takes a CSV file containing PLM embeddings as input, loads a pre-trained Classifier, and makes predictions as to whether or not
each embedded sequence is a truncated selenoprotein.'''

from selenobot.classifiers import Classifier
from selenobot.utils import WEIGHTS_DIR
from selenobot.dataset import Dataset
import os
import numpy as np
import os
import argparse
import torch
import pandas as pd


# TODO: Check the input arguments. 
# TODO: Add some utilities for automatically detecting the latent dimension. 
# TODO: Make a Python API for this script, so that it can be called in the same way. 

def check_args(args:argparse.ArgumentParser):
    pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the input embeddings on which to run the classifier.')
    parser.add_argument('output', help='Path to the file where model predictions will be written.')
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'plm_model_weights.pth'), help='The path to the stored model weights.')
    args = parser.parse_args()

    model = Classifier(latent_dim=1024, hidden_dim=512)
    model.load_state_dict(torch.load(args.weights))
    
    dataset = Dataset(pd.read_csv(args.input)) # Instantiate a Dataset object with the embeddings. 
    reporter = model.predict(dataset)
    predictions = reporter.apply_threshold()

    df = pd.DataFrame({'id':dataset.ids, 'model_output':reporter.outputs, 'prediction':predictions})
    df['seq'] = dataset.seqs # Add sequences to the DataFrame. 
    df.set_index('id').to_csv(args.output)




