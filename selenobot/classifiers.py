'''A binary classification head and associated utilities, designed for handling embedding objects.'''

from tqdm import tqdm
from typing import Optional, NoReturn, Tuple, List

import sys
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional
import sklearn
import warnings

warnings.simplefilter('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainReporter():
    '''A class for managing the results of training a Classifier. The information stored in this
    object can be used to plot training curves.'''

    def __init__(self, epochs:int=None, lr:float=None, batches_per_epoch:int=None):
        '''Initialize a TrainReporter object.
        
        :param epochs: The number of epochs for which the model is trained. 
        :param lr: The learning rate at which the model was trained.
        :batches_per_epoch: The number of batches in each epoch during training. 
        '''
        super().__init__()

        self.epochs = epochs
        self.lr = lr
        self.batches_per_epoch = batches_per_epoch

        # Only store the losses and accuracies, as storing the entire sets of outputs and
        # targets for each batch seems like too much. 
        self.train_losses = []
        self.val_losses = []


    def get_training_curve_data(self) -> pd.DataFrame:
        '''Organizes the data contained in the reporter into a pandas DataFrame to
        make it convenient to plot training curves.'''
        assert len(self.train_losses) == len(self.val_losses), 'reporters.TrainReporter.get_training_curve_data: The number of recorded validation and training losses should be equal.'
        n = len(self.train_losses) # Total number of recorded values. 

        data = {}
        data['metric'] = ['training loss'] * n + ['validation loss'] * n
        data['epoch'] = list(range(n)) * 2
        data['value'] = self.train_losses + self.val_losses

        return pd.DataFrame(data)


class TestReporter():
    '''A class for managing the results of evaluating a Classifier on test data.'''

    def __init__(self, outputs, targets):
        '''Initialize a TestReporter object, which stores data for assessing model 
        performance on a Dataset object.
        
        :param threshold: The threshold to apply to the model outputs when computing predictions.
        '''
        # Make sure outputs and targets are stored as one-dimensional numpy arrays. 
        self.outputs = outputs.detach().numpy().ravel()

        if targets is not None:
            self.targets = targets.detach().numpy().ravel()
        else:
            self.targets = None

        self.loss = None

    def apply_threshold(self, threshold:float=0.5) -> np.array:
        '''Apply the threshold to model outputs.'''
        # Apply the threshold to the output values. 
        outputs = np.ones(self.outputs.shape) # Initialize an array of ones. 
        outputs[np.where(self.outputs < threshold)] = 0
        return outputs

    def get_balanced_accuracy(self, threshold:float=0.5) -> float:
        '''Applies a threshold to the model outputs, and uses it to compute a balanced accuracy score.'''
        outputs = self.apply_threshold(threshold=threshold)
        # Compute balanced accuracy using a builtin sklearn function. 
        return sklearn.metrics.balanced_accuracy_score(self.targets, outputs)

    def get_auc(self) -> float:
        '''Computes the AUC score for the contained outputs and targets.'''
        return sklearn.metrics.roc_auc_score(self.targets, self.outputs)
    
    def get_expected_calibration_error(self, nbins:int=10) -> float:
        '''Calculate the expected calibration error of the test predictions.'''
        # First, sort the predictions into bins. 
        bins = np.linspace(0, 1, num=nbins + 1, endpoint=True)
        # Setting right=False means the lower bound is inclusive, and the upper bound is non-inclusive. 
        bin_idxs = np.digitize(self.outputs, bins, right=False)

        err = 0
        n = len(self.outputs) # Get the total number of outputs. 
        # Should note that bin indices seem to start at 1. 0 is for things that are outside of the bin range. 
        for b in range(1, len(bins)):
            m = sum(bin_idxs == b) # Get the number of things in the bin.

            if m > 0:
                bin_acc = self.targets[bin_idxs == b].mean()
                bin_conf = self.outputs[bin_idxs == b].mean()
                err += (m/n) * abs(bin_acc - bin_conf)
        
        return err


class WeightedBCELoss(torch.nn.Module):
    '''Defining a class for easily working with weighted Binary Cross Entropy loss.'''
    def __init__(self, weight=1):

        super(WeightedBCELoss, self).__init__()
        self.w = weight

    def forward(self, outputs, targets):
        '''Update the internal states keeping track of the loss.'''
        # Make sure the outputs and targets have the same shape.
        outputs = outputs.reshape(targets.shape)
        # reduction specifies the reduction to apply to the output. If 'none', no reduction will be applied, if 'mean,' the weighted mean of the output is taken.
        ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
        # Seems to be generating a weight vector, so that the weight is applied to indices marked with a 1. This
        # should have the effect of increasing the cost of a false negative.
        w = torch.where(targets == 1, self.w, 1).to(DEVICE)

        return (ce * w).mean()


class Classifier(torch.nn.Module):
    '''Class defining the binary classification head.'''

    def __init__(self, 
        hidden_dim:int=512,
        latent_dim:int=1024,
        bce_loss_weight:float=1,
        random_seed:int=42):
        '''
        Initializes a two-layer linear classification head. 

        :param bce_loss_weight: The weight applied to false negatives in the BCE loss function. 
        :param hidden_dim: The number of nodes in the second linear layer of the two-layer classifier.
        :param latent_dim: The dimensionality of the input embedding. 
        '''
        # Initialize the torch Module
        super().__init__()
        torch.manual_seed(random_seed)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid())

        # Initialize model weights according to which activation is used.'''
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        torch.nn.init.xavier_normal_(self.classifier[2].weight)

        self.to(DEVICE)
        self.loss_func = WeightedBCELoss(weight=bce_loss_weight)

    def forward(self, inputs:torch.FloatTensor, batch_size:int=None):
        '''A forward pass of the Classifier.'''
        assert inputs.dtype == torch.float32, f'classifiers.Classifier.forward: Expected input embedding of type torch.float32, not {inputs.dtype}.'
        
        if batch_size is None:  # If no batches are specified, treat as a single batch. 
            outputs = self.classifier(inputs) 
        else:
            outputs = [self.classifier(batch) for batch in torch.split(inputs, batch_size)]
            outputs = torch.concat(outputs)

        return outputs

    def predict(self, dataset) -> TestReporter:
        '''Evaluate the Classifier on the data in the input Dataset. 

        :param dataset: The Dataset object containing the testing data. This Dataset can be either labeled or unlabeled.  
        '''    
        self.eval() # Put the model in evaluation mode. 
        
        inputs, targets = dataset.embeddings, dataset.labels
        outputs = self(inputs, batch_size=32) # Run a forward pass of the model. Batch to limit memory usage. 
            
        reporter = TestReporter(outputs, targets) # Instantiate a Reporter for storing collected data. 

        if targets is not None: # Just in case the dataset is unlabeled. 
            reporter.loss = self.loss_func(outputs, targets).item() # Add the loss to the reporter as a float.

        self.train() # Put the model back in train mode.

        return reporter

    def fit(self, dataloader, val_dataset, epochs:int=10, lr:float=0.01) -> TrainReporter:
        '''Train Classifier model on the data in the DataLoader.

        :param dataloader: The DataLoader object containing the training data. 
        :param val_dataset: The Dataset object containing the validation data. Expected to be a Dataset as defined in datasets.py.
        :param epochs: The number of epochs to train for. 
        :param lr: The learning rate. 
        :param bce_loss_weight: The weight to be passed into the WeightedBCELoss constructor.
        '''
        assert dataloader.dataset.labeled, 'classifiers.Classifier.fit: The input DataLoader must be labeled.'
        assert val_dataset.labeled, 'classifiers.Classifier.fit: The input validation Dataset must be labeled.'

        self.train() # Put the model in train mode.

        reporter = TrainReporter(epochs=epochs, lr=lr, batches_per_epoch=len(dataloader))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc='classifiers.Classifier.fit')

        # Want to log the initial training and validation metrics.
        val_loss = self.predict(val_dataset).loss
        train_loss = self.predict(dataloader.dataset).loss

        reporter.val_losses.append(val_loss)
        reporter.train_losses.append(train_loss)

        # Set the progress bar values. 
        pbar.set_postfix({'val_loss':np.round(val_loss, 4), 'train_loss':np.round(train_loss, 4)})
        for _ in pbar:
            train_losses = [] # Accumulate the train loss over the epoch. 
            for batch in dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['embedding']), batch['label']
                train_loss = self.loss_func(outputs, targets)

                # Accumulate the results over batches.
                train_losses.append(train_loss.item())

                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Evaluate the the train loss and train accuracy on the validation set and add to the reporter. 
            val_loss = self.predict(val_dataset).loss
            reporter.val_losses.append(val_loss)
            reporter.train_losses.append(np.mean(train_losses))

            # Update the progress bar. 
            pbar.set_postfix({'val_loss':np.round(val_loss, 4), 'train_loss':np.round(np.mean(train_losses), 4)})
            
        pbar.close()
        
        return reporter


class SimpleClassifier(Classifier):
    '''Class defining a simplified version of the binary classification head.'''

    def __init__(self, 
        bce_loss_weight:float=1.0,
        latent_dim:int=1,
        random_seed:int=42):
        '''Initializes a single-layer linear classification head.'''

        # Initialize the torch Module. The classifier attribute will be overridden by the rest of this init function. 
        super().__init__()
        torch.manual_seed(random_seed)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1),
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
#         model = Classifier(hidden_dim=512, latent_dim=1024, bce_loss_weight=bce_loss_weight) # Reset the model. 
#         model.fit(dataloader, val_dataset, epochs=epochs, lr=lr, bce_loss_weight=bce_loss_weight)
        
#         # Evaluate the performance of the fitted model on the data.
#         reporter = model.predict(val_dataset)
#         return -reporter.get_balanced_accuracy() # Return negative so as not to minimize accuracy.
    
#     result = skopt.gp_minimize(objective, search_space)
#     return result.x





 





