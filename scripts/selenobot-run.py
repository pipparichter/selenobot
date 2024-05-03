'''A script for running a trained classifier on embedded genome data. This script should be run on the HPC, which is where the 
embedding and annotation data is stored.'''

from selenobot.classifiers import Classifier
from selenobot.utils import WEIGHTS_DIR
from selenobot.dataset import Dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import h5py
import pickle
import numpy as np
import glob # For file pattern matching.
import os
import Bio.SeqIO as SeqIO
from scipy.stats import fisher_exact
from datetime import date
import argparse


DATE = date.today().strftime('%d.%m.%y')
ANNOTATION_DIR = '/groups/fischergroup/goldford/gtdb/ko_annotations/' # Path to KO annotations on HPC.
EMBEDDING_DIR = '/groups/fischergroup/goldford/gtdb/embedding/' # Path to embeddings on HPC.

# def load_embedding(path:str):
#     '''Load PLM embeddings from a file at the specified path. Each file contains embedded sequences for a
#     single genome, stored as an HDF file.'''
#     f = h5py.File(file_name)
#     df = []
#     gene_ids = list(f.keys())
#     for key in gene_ids:
#         data.append(f[key][()]) # What is this doing?
#     f.close()
#     df = pd.DataFrame(np.asmatrix(df), index=gene_ids)
#     return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the input embeddings on which to run the classifier.')
    parser.add_argument('output', help='Path to the file where model predictions will be written.')
    parser.add_argument('--genome-id', default=None, type=str, help='Genome ID of the organism for which predictions are being generated.')
    # parser.add_argument('--add-annotations', default=0, type=bool, default=0)
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'plm_model_weights.pth'), help='The path to the stored model weights.')
    args = parser.parse_args()

    model = Classifier(latent_dim=1024, hidden_dim=512)
    model.load_state_dict(torch.load(args.weights))
    
    dataset = Dataset(pd.read_csv(args.input)) # Instantiate a Dataset object with the embeddings. 
    reporter = model.predict(dataset)
    predictions = reporter.apply_threshold()
    df = pd.DataFrame({'id':dataset.ids, 'confidence':reporter.outputs, 'prediction':predictions})
    df['seq'] = dataset.seqs # Add sequences to the DataFrame. 
    df.set_index('id').to_csv(args.output)




