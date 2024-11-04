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
from sklearn.preprocessing import StandardScaler

# warnings.simplefilter('ignore')

class Unpickler(pickle.Unpickler):
    '''https://github.com/pytorch/pytorch/issues/16797'''
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        else: return super().find_class(module, name)


class WeightedBCELoss(torch.nn.Module):
    '''Defining a class for easily working with weighted Binary Cross Entropy loss.'''
    def __init__(self):

        super(WeightedBCELoss, self).__init__()
        self.w1 = 1
        self.w0 = 1

    def fit(self, dataset):
        '''Compute the weights to use based on the frequencies of each class in the input Dataset. 
        Formula used to compute weights is from this: 
        https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75'''

        n = len(dataset)
        n1 = dataset.labels.sum()
        n0 = n - n1

        self.w1 = n / (n1  * 2) 
        self.w0 = n / (n0  * 2) 

    def forward(self, outputs, targets):
        '''Update the internal states keeping track of the loss.'''
        # Make sure the outputs and targets have the same shape.
        # outputs = outputs.reshape(targets.shape)
        outputs = outputs.view(targets.shape)
        # reduction specifies the reduction to apply to the output. If 'none', no reduction will be applied, if 'mean,' the weighted mean of the output is taken.
        # ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
        # NOTE: Switch to with_logits for numerical stability, see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 
        # ce = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
        # Generate a weight vector using self.w1 and self.w0. 
        w = torch.where(targets == 1, self.w1, self.w0) # .to(DEVICE)

        return (ce * w).mean()


class Classifier(torch.nn.Module):
    '''Class defining the binary classification head.'''

    attrs = ['epochs', 'batch_size', 'lr', 'val_accs', 'train_losses', 'best_epoch']
    params = ['hidden_dim', 'input_dim', 'standardize', 'half_precision']

    def __init__(self, 
        hidden_dim:int=512,
        input_dim:int=1024,
        half_precision:bool=False, 
        scale:bool=True):
        '''
        Initializes a two-layer linear classification head. 

        :param bce_loss_weight: The weight applied to false negatives in the BCE loss function. 
        :param hidden_dim: The number of nodes in the second linear layer of the two-layer classifier.
        :param input_dim: The dimensionality of the input embedding. 
        '''
        # Initialize the torch Module
        super().__init__()

        self.dtype = torch.float16 if half_precision else torch.float32

        self.half_precision = half_precision
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        # self.bce_loss_weight = bce_loss_weight 

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_dim, 1, dtype=self.dtype),
            torch.nn.Sigmoid())

        # Initialize model weights according to which activation is used.'''
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        torch.nn.init.xavier_normal_(self.classifier[2].weight)

        # self.loss_func = torch.nn.functional.binary_cross_entropy_with_logits
        self.loss_func = WeightedBCELoss()

        # Parameters to be populated when the model has been fitted. 
        self.best_epoch = None
        self.epochs = None
        self.lr = None 
        self.batch_size = None
        self.train_losses, self.val_accs = None, None

        self.instances_seen_during_training = 0
        
        self.scaler = StandardScaler() if scale else None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier.to(self.device)
        self.to(self.device)

    # TODO: Do I still need the batch size parameter here?
    def forward(self, inputs:torch.FloatTensor, low_memory:bool=True):
        '''A forward pass of the Classifier.'''
        print('input device:', inputs.get_device())
        for layer in self.classifier.layers:
            print('layer weights device:', layer.weight.get_device())
        if low_memory:
            batch_size = 32
            outputs = [self.classifier(batch) for batch in torch.split(inputs, batch_size)]
            return torch.concat(outputs)
        else:
            return self.classifier(inputs) 

    def predict(self, dataset, threshold:float=0.5) -> np.ndarray:
        '''Evaluate the Classifier on the data in the input Dataset.'''   
        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        if self.scaler is not None:
            dataset.scale(self.scaler)
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(dataset.embeddings) # Run a forward pass of the model. Batch to limit memory usage.
            # Apply sigmoid activation, which is usually applied as a part of the loss function. 
            # outputs = torch.nn.functional.sigmoid(outputs).ravel()
            outputs = outputs.cpu().numpy()

            if threshold is not None: # Apply the threshold to the output values. 
                outputs_with_threshold = np.ones(outputs.shape) # Initialize an array of ones. 
                outputs_with_threshold[np.where(outputs < threshold)] = 0
                return outputs_with_threshold
            else:
                return outputs


    # def _loss(self, dataset):
    #     outputs, targets = self(dataset.get_embeddings()), dataset.labels 
    #     return self.loss_func(outputs, targets)

    def fit(self, train_dataset, val_dataset, epochs:int=10, lr:float=0.01, batch_size:int=16, early_stopping:bool=True, balance_batches:bool=True, weighted_loss:bool=False):
        '''Train Classifier model on the data in the DataLoader.

        :param train_dataset: The Dataset object containing the training data. 
        :param val_dataset: The Dataset object containing the validation data.
        :param epochs: The maximum number of epochs to train for. 
        :param lr: The learning rate. 
        :param batch_size: The size of the batches to use for model training.
        '''
        self.train() # Put the model in train mode.
        print(f'Classifier.fit: Training on device {self.device}.')

        if self.scaler is not None:
            self.scaler.fit(train_dataset.embeddings.cpu().numpy()) # Fit the scaler on the training dataset. 
            train_dataset.scale(self.scaler)
            # val_dataset.scale(self.scaler)
        if weighted_loss:
            self.loss_func.fit(train_dataset) # Set the weights of the loss function.

        # NOTE: What does the epsilon parameter do?
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4 if self.half_precision else 1e-8)

        best_epoch, best_model_weights = 0, copy.deepcopy(self.state_dict())

        # Want to log the initial training and validation metrics. 
        val_accs = [balanced_accuracy_score(val_dataset.labels.cpu().numpy(), self.predict(val_dataset))]
        train_losses = [self.loss_func(self(train_dataset.embeddings).ravel(), train_dataset.labels).item()]

        dataloader = get_dataloader(train_dataset, batch_size=batch_size, balance_batches=balance_batches)
        pbar = tqdm(total=epochs * len(dataloader), desc=f'Classifier.fit: Training classifier, epoch 0/{epochs}.') # Make sure the progress bar updates for each batch. 

        for epoch in range(epochs):
            train_loss = []
            # for batch in tqdm(dataloader, desc='Classifier.fit: Processing batches...'):
            for batch in dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['embedding'], low_memory=False), batch['label'] # Takes about 30 percent of total batch time. 
                loss = self.loss_func(outputs.ravel(), targets)
                loss.backward() # Takes about 10 percent of total batch time. 
                train_loss.append(loss.item()) # Store the batch loss to compute training loss across the epoch. 
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1) # Update progress bar after each batch. 

                # Keep track of the number of data points the model "sees" during training. 
                self.instances_seen_during_training += batch_size
            
            train_losses.append(np.mean(train_loss))
            val_accs.append(balanced_accuracy_score(val_dataset.labels.cpu().numpy(), self.predict(val_dataset)))
            
            pbar.set_description(f'Classifier.fit: Training classifier, epoch {epoch}/{epochs}. Validation accuracy {np.round(val_accs[-1], 2)}')

            if val_accs[-1] > min(val_accs[:-1]):
                best_epoch = epoch
                best_model_weights = copy.deepcopy(self.state_dict())

        if early_stopping:
            print(f'Classifier.fit: Loading best model weights, encountered at epoch {best_epoch}.')
            self.load_state_dict(best_model_weights) # Load the best model weights. 

        # Save training values in the model. 
        self.best_epoch = best_epoch
        self.val_accs = val_accs # Don't include the initializing np.inf loss. 
        self.train_losses = train_losses
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path:str):
        with open(path, 'rb') as f:
            # obj = pickle.load(f)
            obj = Unpickler(f).load()
        return obj    



class SimpleClassifier(Classifier):
    '''Class defining a simplified version of the binary classification head.'''

    def __init__(self, 
        bce_loss_weight:float=1.0,
        input_dim:int=1,
        random_seed:int=42):
        '''Initializes a single-layer linear classification head.'''

        # Initialize the torch Module. The classifier attribute will be overridden by the rest of this init function. 
        super().__init__()
        torch.manual_seed(random_seed)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Sigmoid())

        # Initialize model weights according to which activation is used.
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        self.to(DEVICE)

        self.loss_func = WeightedBCELoss(weight=bce_loss_weight)




# def optimize(dataloader, val_dataset:Dataset, n_calls:int=50): 

#     # Probably keep the weights as integers, at least for now. 
#     search_space = [skopt.space.Real(1, 10, name='bce_loss_weight')]

#     lr = 0.001
#     epochs = 10

#     # Create the objective function. Decorator allows named arguments to be inferred from search space. 
#     @skopt.utils.use_named_args(dimensions=search_space)
#     def objective(bce_loss_weight=None):
#         model = Classifier(hidden_dim=512, input_dim=1024, bce_loss_weight=bce_loss_weight) # Reset the model. 
#         model.fit(dataloader, val_dataset, epochs=epochs, lr=lr, bce_loss_weight=bce_loss_weight)
        
#         # Evaluate the performance of the fitted model on the data.
#         info = model.predict(val_dataset)
#         return -info.get_balanced_accuracy() # Return negative so as not to minimize accuracy.
    
#     result = skopt.gp_minimize(objective, search_space)
#     return result.x


    # def save(self, path:str):
    #     info = dict()
    #     for attr in Classifier.attrs:
    #         info[attr] = getattr(self, attr)
    #     for param in Classifier.params:
    #         info[param] = getattr(self, param)

    #     info['state_dict'] = self.state_dict() #.numpy()
    #     # Save information for re-loading the scaler. 
    #     info['scaler_mean'] = self.scaler.mean_
    #     info['scaler_scale'] = self.scaler.scale_

    #     with open(path, 'w') as f:
    #         json.dump(info, f, cls=NumpyEncoder)

    # @classmethod
    # def load(cls, path:str):
    #     with open(path, 'r') as f:
    #         info = json.load(f)
        
    #     params = {param:info.get(param) for param in Classifier.params}
    #     obj = cls(**params) # Initialize a new object with the stored parameters. 

    #     state_dict = {k:torch.Tensor(v) for k, v in info['state_dict'].items()}
    #     state_dict = {k:v.to(torch.float16) if obj.half_precision else v for k, v in state_dict.items()} # Convert to half-precision if specified. 
    #     obj.load_state_dict(state_dict) # Load the saved state dict. NOTE: Might need to convert to a tensor. 
        
    #     # Set all the other model parameters. 
    #     for attr in Classifier.attrs:
    #         setattr(obj, attr, info.get(attr))

    #     # Load in the values from the fitted scaler, if the saved model had been normalized. 
    #     if obj.standardize:
    #         obj.scaler.mean_ = np.array(info.get('scaler_mean'))
    #         obj.scaler.scale_ = np.array(info.get('scaler_scale'))
        
    #     return obj




 





