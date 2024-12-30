'''A binary classification head and associated utilities, designed for handling embedding objects.'''

from tqdm import tqdm
from typing import Optional, NoReturn, Tuple, List
from selenobot.datasets import get_dataloader
import sys
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional
import sklearn
# import json
import time
from sklearn.metrics import balanced_accuracy_score
from selenobot.utils import NumpyEncoder
import warnings
import copy
import io
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from selenobot.tools import CDHIT, MUSCLE
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord


# warnings.simplefilter('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Unpickler(pickle.Unpickler):
    '''https://github.com/pytorch/pytorch/issues/16797'''
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        else: return super().find_class(module, name)



# TODO: Confirm equivalence beween using pytorch binary cross-entropy and what I am doing here. 
class WeightedCrossEntropyLoss(torch.nn.Module):
    '''Defining a class for easily working with weighted Cross Entropy loss.'''
    def __init__(self, n_classes:int=2, half_precision:bool=False):

        super(WeightedCrossEntropyLoss, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if half_precision else torch.float32
        self.weights = torch.Tensor([1] * n_classes).to(self.dtype).to(self.device)
        self.n_classes = n_classes

        self.to(self.device) # Not actually sure if this is necessary. 

    def fit(self, dataset):
        '''Compute the weights to use based on the inverse frequencies of each class. '''

        N = len(dataset)
        n = [(dataset.labels == i).sum() for i in range(self.n_classes)]
        # NOTE: I wonder if I should be scaling this by the number of classes, so that more classes
        # doesn't result in larger weights? I am going to to keep things between 0 and 1. 
        self.weights = torch.Tensor([(N / (n_i * self.n_classes)) for n_i in n]).to(self.dtype).to(self.device)


    # NOTE: Targets can be class indices, as opposed to class probabilities. Should decide which one to use. 
    def forward(self, outputs, targets):
        '''Compute the weighted loss between the targets and outputs. 

        :param outputs: A Tensor of size (batch_size, n_classes). All values should be between 0 and 1, and sum to 1. 
        :param targets: A Tensor of size (batch_size, n_classes). All values should be 0 or 1.  
        '''
        outputs = outputs.view(targets.shape) # Make sure the outputs and targets have the same shape. Use view to avoid copying. 
        # Reduction specifies the reduction to apply to the output. If 'none', no reduction will be applied, if 'mean,' the weighted mean of the output is taken.
        ce = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')

        w = torch.unsqueeze(self.weights, 0).repeat(len(outputs), 1)
        w = (targets * w).sum(axis=1)

        return (ce * w).mean()

# TODO: Read about how Gaussian HMMs work (where does the Gaussian part come in?)
# TODO: Do I want to use a non-Gaussian emission probability?
# TODO: Do I want a different HMM for different clusters in each category? And possibly base them on multi-sequence alignments. 

class HMM():
    
    def __init__(self, n_classes:int=2, half_precision:bool=False, models_dir:str='../models/'):
        self.models = {i:[] for i in range(n_classes)} 
        self.msas = {i:[] for i in range(n_classes)}

        self.models_dir = models_dir
        # Apparently no way to create a custom alphabet, so id it id a problem I will replace non-standard AAs. 
        self.alphabet = pyhmmer.easel.Alphabet.amino() # There are 20 regular amino acids and 9 extra symbols in this alphabet. 

    def align(self, df:pd.DataFrame, label:int=None, cluster:int=None):
        
        name = f'{label}_{cluster}'
        alignment_path = MUSCLE(df, name=name, cwd=self.models_dir).run()
        self.msas[label].append(alignment_path)
        
    # I don't need a validation dataset here... should I combine, or ignore?
    def fit(self, train_dataset, val_dataset): 
        
        n_msas = 0
        for label, df in tqdm(train_dataset.metadata.groupby('label'), desc='HMM.fit: Generating multi-sequence alignments...'):
            # First, cluster the training dataset at 50 percent sequence similarity. 
            cdhit = CDHIT(df, name='hmm', c_cluster=0.5, cwd=self.models_dir)
            # Sequences have already been de-replicated, so don't do that again. 
            df = cdhit.run(overwrite=False, dereplicate=False)
            # Generate a MSA for each cluster. 
            for cluster, cluster_df in df.groupby(cluster):
                self.align(cluster_df, label=label, cluster=cluster)
                n_msas += 1

        pbar = tqdm(total=n_msas, desc='HMM.fit: Building HMMs...')
        for label, alignment_paths in self.msas.items():
            for alignment_path in alignment_paths:

                msa = phmmer.easel.MSAFile(alignment_path, digital=True, alphabet=self.alphabet)
                msa.name = os.path.basename(alignment_path).replace('.afa', '') # Need to set this for the HMM. 
                builder = pyhmmer.plan7.Builder(self.alphabet)
                background = pyhmmer.plan7.Background(self.alphabet)

                # What are the other outputs?
                model, _, _ = builder.build_msa(msa, self.alphabet)
                self.models[label].append(model)
                pbar.update(1)
    
    def predict(self, dataset):

        seqs = dataset.metadata.seq.values
        # Need to digitize the hits  so they work with the HMMs. 
        queries = [pyhmmer.easel.DigitalSequence(self.alphabet, sequence=seq) for seq in seqs]

        for label, hmms in self.models.items():
            # I think this is a list of TopHits objects, but not completely sure. 
            tophits = pyhmmer.hmmer.hmmscan(queries, hmms)


class NN(torch.nn.Module):

    def __init__(self, hidden_dim:int=512, input_dim:int=1024, output_dim:int=2, half_precision:bool=False):

        super(NN, self).__init__()
       
        self.dtype = torch.bfloat16 if half_precision else torch.float32

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, dtype=self.dtype))
            # torch.nn.Softmax(dim=1))
        # Initialize model weights according to which activation is used. See https://www.pinecone.io/learn/weight-initialization/#Summing-Up 
        torch.nn.init.kaiming_normal_(self.model[0].weight)
        # torch.nn.init.xavier_normal_(self.classifier[2].weight)

        self.loss_func = WeightedCrossEntropyLoss(half_precision=half_precision, n_classes=output_dim)

        self.instances_seen_during_training = 0
        self.scaler = StandardScaler()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)


    # TODO: Do I still need the batch size parameter here?
    def forward(self, inputs:torch.FloatTensor):
        '''A forward pass of the Classifier.
        
        :param inputs:
        :param low_memory
        '''
        self.to(device)
        return self.model(inputs) 


    def predict(self, dataset) -> pd.DataFrame:
        '''Evaluate the Classifier on the data in the input Dataset.'''   
        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        dataset = dataset.scale(self.scaler)
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(dataset.embeddings) # Run a forward pass of the model. Batch to limit memory usage.
            # Apply sigmoid activation, which is usually applied as a part of the loss function. 
            outputs = torch.nn.functional.softmax(outputs, 1)
            outputs = outputs.cpu().numpy()

            # Organize the predictions into a DataFrame.
            predictions = pd.DataFrame(index=dataset.ids)
            # for i, category in dataset.categories.items():
            for i in range(outputs.shape[-1]):
                predictions[f'probability_{dataset.categories[i]}'] = outputs[:, i].ravel()
            predictions['prediction'] = np.argmax(outputs, axis=1).ravel() # Convert out of one-hot encodings.

            return predictions


    def accuracy(self, dataset) -> float:
        '''Compute the balanced accuracy of the model on the input dataset.'''
        labels = dataset.labels.cpu().numpy().ravel() # Get the non-one-hot encoded labels from the dataset. 
        predictions = self.predict(dataset).prediction.values.ravel()
        return balanced_accuracy_score(labels, predictions)


    def fit(self, train_dataset, val_dataset, epochs:int=10, lr:float=1e-8, batch_size:int=16, balance_batches:bool=True, weighted_loss:bool=False):
        '''Train Classifier model on the data in the DataLoader.

        :param train_dataset: The Dataset object containing the training data. 
        :param val_dataset: The Dataset object containing the validation data.
        :param epochs: The maximum number of epochs to train for. 
        :param lr: The learning rate. 
        :param batch_size: The size of the batches to use for model training.
        '''
        self.train() # Put the model in train mode.
        print(f'Classifier.fit: Training on device {self.device}.')

        self.scaler.fit(train_dataset.embeddings.cpu().numpy()) # Fit the scaler on the training dataset. 
        train_dataset = train_dataset.scale(self.scaler) # NOTE: Don't need to scale the validation dataset, as this is done by predict. 

        if weighted_loss: # Loss function has equal weights by default.
            self.loss_func.fit(train_dataset) # Set the weights of the loss function.

        # NOTE: What does the epsilon parameter do?
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        best_epoch, best_model_weights = 0, copy.deepcopy(self.state_dict())

        # Want to log the initial training and validation metrics. 
        self.val_accs = [self.accuracy(val_dataset)]
        self.train_losses = []

        dataloader = get_dataloader(train_dataset, batch_size=batch_size, balance_batches=balance_batches)
        pbar = tqdm(total=epochs * len(dataloader), desc=f'Classifier.fit: Training classifier, epoch 0/{epochs}. Validation accuracy {np.round(self.val_accs[-1], 2)}') 

        for epoch in range(epochs):
            epoch_train_loss = []

            for batch in dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['embedding']), batch['label_one_hot_encoded'] 
                loss = self.loss_func(outputs, targets)
                loss.backward() # Takes about 10 percent of total batch time. 
                epoch_train_loss += [loss.item()]
                
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.update(1) # Update progress bar after each batch. 
            
            self.val_accs += [self.accuracy(val_dataset)]
            self.train_losses += [np.mean(epoch_train_loss)]
            
            pbar.set_description(f'Classifier.fit: Training classifier, epoch {epoch}/{epochs}. Validation accuracy {np.round(self.val_accs[-1], 2)}')

            if self.val_accs[-1] > max(self.val_accs[:-1]):
                best_epoch = epoch
                best_model_weights = copy.deepcopy(self.state_dict())

        print(f'Classifier.fit: Loading best model weights, encountered at epoch {best_epoch}.')
        self.load_state_dict(best_model_weights) # Load the best model weights. 

        # Save training parameters in the model. 
        self.best_epoch = best_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr



class Classifier():

    model_types = ['hmm', 'nn']

    def __init__(self, n_classes:int=2, model_type:str='nn', **kwargs):
        
        self.model = HMM(n_classes=n_classes, **kwargs) if model_type == 'hmm' else NN(output_dim=n_classes, **kwargs)

    def fit(self, train_dataset, val_dataset, **kwargs):
        self.model.fit(train_dataset, val_dataset, **kwargs)

    def predict(self, dataset) -> pd.DataFrame:
        return self.model.predict(dataset)

    @classmethod
    def load(cls, path:str):
        with open(path, 'rb') as f:
            # obj = pickle.load(f)
            obj = Unpickler(f).load()
        return obj    

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)



