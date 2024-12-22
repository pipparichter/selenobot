import pandas as pd
import os
import numpy as np
from selenobot.datasets import Dataset, get_dataloader
from selenobot.classifiers import *
from selenobot.files import *
from selenobot.embedders import embed

# Define some important directories...
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results') # Get the path where results are stored.
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Get the path where results are stored. 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts') # Get the path where results are stored.

# f = NcbiXmlFile(os.path.join(DATA_DIR, 'uniprot_sprot.xml'))
# f.dataframe().to_csv(os.path.join(DATA_DIR, 'uniprot_sprot.csv'))

# df = pd.read_csv(os.path.join(DATA_DIR, 'uniprot_truncated.csv'))
# embed(df, os.path.join(DATA_DIR, 'uniprot_truncated.h5'))

# Remove the aa_4mer embeddings from the validation and testing datasets. 
for file_name in ['test.h5', 'val.h5']:
    store = pd.HDFStore(file_name)
    store.remove('aa_4mer')
    store.close()
    print(f'Removed aa_4mer embeddings from {file_name}.')