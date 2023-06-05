

from dataset import SequenceDataset
from esm import ESMClassifier, esm_train, esm_test
# from logreg import LogisticRegressionClassifier, logreg_train, logreg_test
import scipy.stats as stats
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Apparently this frees up some memory. 
torch.cuda.empty_cache()

train_data = SequenceDataset(pd.read_csv('./train.csv'))
test_data = SequenceDataset(pd.read_csv('./test.csv'))

model_esm = ESMClassifier()

losses_esm = esm_train(model_esm, train_data, test_data=test_data, batch_size=5, n_epochs=500)
torch.save(model_esm, 'model_esm.pickle')

with open('./losses_esm.txt', 'w') as f:
    f.write(str(losses_esm))