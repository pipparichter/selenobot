'''
A script to be run on the Google Cloud VM, for training the ESM classifier. 
'''

from src.dataset import SequenceDataset
from src.esm import ESMClassifier, esm_train, esm_test
# from logreg import LogisticRegressionClassifier, logreg_train, logreg_test
import scipy.stats as stats
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# Apparently this frees up some memory. 
# torch.cuda.empty_cache()

# train_data = SequenceDataset(pd.read_csv('./data/train.csv'), sec_fasta_path='./data/sec.fasta')
# test_data = SequenceDataset(pd.read_csv('./data/test.csv'), sec_fasta_path='./data/sec.fasta')

# train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
# test_loader = DataLoader(train_data, shuffle=False, batch_size=32)

# Instantiate the classifier. 
model_esm = ESMClassifier()
for module in ESMClassifier().modules():
    print(type(module))

# losses_esm = esm_train(model_esm, train_loader, test_loader=test_loader, n_epochs=100)
# torch.save(model_esm, 'model_esm.pickle')

# with open('./losses_esm.txt', 'w') as f:
#     f.write(str(losses_esm))