'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''

import transformers # Import the pretrained models from HuggingFace.
from transformers import EsmForSequenceClassification, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
# import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
# from transformers import AdamW
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from torch.optim import Adam
# from peft import get_peft_model, LoraConfig 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.nn.functional import cross_entropy, binary_cross_entropy

# NOTE: ESM has a thing which, by default, adds a linear layer for classification. Might be worth 
# training my own version of this layer in the future. 


class ESMClassifier(torch.nn.Module):
    
    def __init__(self, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifier object, as well as the torch.nn.Module superclass. 
        
        kwargs:

        '''
        # Initialize the super class, torch.nn.Module. 
        super(ESMClassifier, self).__init__()

        self.model = EsmForSequenceClassification.from_pretrained(name, num_labels=1)

        # Freeze all model weights which aren't related to the classifier. 
        for name, param in self.model.esm.named_parameters():
            param.requires_grad = False
    
    # TODO: Print out some kind of summary of the model, with trainable and non-trainable params. 
    def summary(self):

        table = PrettyTable(['name', 'num_params', 'fixed'])

        num_fixed = 0
        num_total = 0

        params = {}

        for name, param in self.named_parameters():
            num_params = param.numel()
            fixed = str(not param.requires_grad)
            table.add_row([name, num_params, fixed])

            if not param.requires_grad:
                num_fixed += num_params
            
            num_total += num_params
        
        print(table)
        print('TOTAL:', num_total)
        print('TRAINABLE:', num_total - num_fixed, f'({int(100 * (num_total - num_fixed)/num_total)}%)')


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        '''

        kwargs:
            - input_ids (torch.Tensor): Has a shape of (batch_size, sequence length). I think these are just the tokenized
                equivalents of each sequence element. 
            - attention_mask (torch.Tensor): A tensor of ones and zeros. 
            - labels (torch.Tensor): Has a size of (batch_size,). Indicated whether or not a sequence is truncated (1)
                or full-length (0).
        
        returns: transformers.modeling_outputs.SequenceClassifierOutput
        '''
        # if self.use_builtin_classifier:
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        # Normalize the predictions. 
        logits = torch.nn.functional.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = binary_cross_entropy(logits, labels)
        
        return logits, loss


# TODO: This is code duplication. Probably should come up with a way to organize functions. 
# Also kind of reluctant to put this in utils, because that's mostly file reading and writing.  
def esm_train(model, train_loader, test_loader=None, n_epochs=300):
    '''

    args:
        - model (torch.nn.Module)
        - train_loader (torch.utils.data.DataLoader)
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 

    # losses = {'train':[], 'test':[], 'accuracy':[]}
    losses = {'train':[], 'test':[]}

    optimizer = Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train() # Put the model in training mode. 

    for epoch in tqdm(range(n_epochs), desc='Training classifier...'):
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 

            logits, loss = model(**batch)
            loss.backward() #
            optimizer.step() 
            optimizer.zero_grad()
        
        losses['train'].append(loss.item()) # Add losses to a history. 
        
        if test_loader is not None:
            # test_loss, accuracy = esm_test(model, test_loader)
            test_loss = esm_test(model, test_loader)
            losses['test'].append(test_loss.item())
            # losses['accuracy'].append(accuracy.item())
            # Make sure to put the model back in train mode. 
            model.train()

    return losses


def esm_test(model, test_loader):
    '''
    Evaluate the model on the test data. 
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 
    
    model.eval() # Put the model in evaluation mode. 

    logits, loss = [], [] # Need to do this in batches of one. 
    for batch in tqdm(test_loader, desc='Calculating batch loss...'): 
        batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 
        
        with torch.no_grad():
            batch_logits, batch_loss = model(**batch)
        loss.append(batch_loss.expand(1))
        logits.append(batch_logits)

    # Concatenate the accumulated results. 
    logits = torch.cat(logits)
    loss = torch.mean(torch.cat(loss))

    # In addition to the loss, get the accuracy.
    prediction = torch.round(logits).to(device) # To zero or one.
    # accuracy = (prediction == test_loader.labels.to(device)).float().mean() # Should be able to do this if I don't shuffle. 

    # return loss, accuracy
    return loss
