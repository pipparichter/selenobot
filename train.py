'''Small script which loads the training data, trains the input model, pickles the resulting reporter object, and saves the model weights. This must be run outside
the Jupyter notebook, or the kernel crashes.'''
from src.classifiers import Classifier, SimpleClassifier
from src.dataset import get_dataloader, Dataset
import os
import pandas as pd
import pickle
import numpy as np
import torch

# data = pd.read_csv(test_path)
# data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
# data.to_csv(test_path, index=False)

# Define some key directories. 
MAIN_DIR = '/home/prichter/Documents/selenobot/'
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/detect/'
TRAIN_PATH = os.path.join(DETECT_DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DETECT_DATA_DIR, 'val.csv')


def train(
    embedder=None,
    epochs=None,
    latent_dim=None,
    hidden_dim=None,
    selenoprotein_fraction:float=None,
    model_weights_path:str=None,
    reporter_path:str=None,
    simple_classifier:bool=False):
    '''The main body of the program.'''


    train_dataloader = get_dataloader(TRAIN_PATH, balance_batches=True, selenoprotein_fraction=selenoprotein_fraction, batch_size=1024, embedder=embedder)
    if simple_classifier:
        model = SimpleClassifier(latent_dim=latent_dim)
    else:
        model = Classifier(latent_dim=latent_dim, hidden_dim=hidden_dim)
    
    train_reporter = model.fit(train_dataloader, val_dataset=Dataset(pd.read_csv(VAL_PATH), embedder=embedder), epochs=epochs, lr=0.001)
    # Save the reporter and model weights. 
    with open(reporter_path, 'wb') as f:
        pickle.dump(train_reporter, f)
    torch.save(model.state_dict(), model_weights_path)


if __name__ == '__main__':

    aac_kwargs = {'embedder':'aac', 'latent_dim':21, 'hidden_dim':8, 'epochs':10}
    length_kwargs = {'embedder':'length', 'latent_dim':1, 'hidden_dim':8, 'epochs':10}
    plm_kwargs = {'embedder':'plm', 'latent_dim':1024, 'hidden_dim':512, 'epochs':10}

    for kwargs in [aac_kwargs, length_kwargs, plm_kwargs]:
        kwargs['selenoprotein_fraction'] = 0.5
        kwargs['reporter_path'] = os.path.join(MAIN_DIR, kwargs['embedder'] + '_train_reporter.pkl')
        kwargs['model_weights_path'] = os.path.join(MAIN_DIR, kwargs['embedder'] + '_model_weights.pth')
        train(**kwargs)

    # for selenoprotein_fraction in np.arange(0.1, 0.6, 0.1):
    #     aac_kwargs['selenoprotein_fraction'] = selenoprotein_fraction
    #     aac_kwargs['reporter_path'] = os.path.join(MAIN_DIR, 'aac_train_reporter_selenoprotein_fraction={selenoprotein_fraction}.pkl')
    #     aac_kwargs['model_weights_path'] = os.path.join(MAIN_DIR, 'aac_model_weights_selenoprotein_fraction={selenoprotein_fraction}.pth')
    #     main(**aac_kwargs)

    # aac_kwargs['reporter_path'] = os.path.join(MAIN_DIR, 'aac_train_reporter_unbalanced.pkl')
    # aac_kwargs['model_weights_path'] = os.path.join(MAIN_DIR, 'aac_model_weights_selenoprotein_unbalanced.pth')
    # main(**aac_kwargs)

