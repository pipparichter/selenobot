from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os


# Define some important directories...
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results') # Get the path where results are stored.
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Get the path where results are stored. 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts') # Get the path where results are stored.


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='plm', type=str, help='The type of embedding to load from the Dataset.')
    parser.add_argument('--input-path', help='The path to the embeddings for which to generate predictions.')
    parser.add_argument('--model-name', default='model_epochs_1000_lr_e8.pkl', help='The name of the model to load.')

    args = parser.parse_args()

    dataset = Dataset.from_hdf(args)
    

