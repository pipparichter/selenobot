'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''
import transformers # Import the pretrained models from HuggingFace.
from transformers import EsmForSequenceClassification, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.nn.functional import cross_entropy, binary_cross_entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EsmClassifierV2(torch.nn.Module):
    
    def __init__(self, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifierV2 object, as well as the torch.nn.Module superclass. 
        
        args:
            - name (str): The name of the pretrained model to use. 
        '''
        # Initialize the super class, torch.nn.Module. 
        super(EsmClassifierV2, self).__init__()

        # I don't think there is a way to turn off the pooling layer when loading from pretrained. 
        # For now, just ignore it. 
        self.esm = EsmModel.from_pretrained(name)

        # Should be 320... 
        hidden_size = self.esm.get_submodule('encoder.layer.5.output.dense').out_features

        # TODO Possibly add a couple of layers, because I think hidden_size is large.
        self.classifier = torch.nn.Linear(hidden_size, 1)

        # Freeze all esm model weights. 
        for name, param in self.esm.named_parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, index=None, embeddings=None):
        '''
        A forward pass of the EsmClassifierV2. 

        args:
            - input_ids (torch.Tensor): Has a shape of (batch_size, sequence length). The tokenized
                equivalents of each sequence element. 
            - attention_mask (torch.Tensor): A tensor of ones and zeros. 
            - labels (torch.Tensor): Has a size of (batch_size,). Indicates whether or not a sequence 
                is a selenoprotein (1) or not (0).
            - index (?): The indices corresponding to the batch data's original position in the dataset. 
            - embeddings (?): The ESM-generated embedding of the amino acid sequence. 
        '''
        if embeddings is not None:
            # Make sure data types line up, so the linear layer doesn't flip out. 
            embedding = embeddings.to(self.classifier.weight.dtype)
        else: # Only bother doing a forward pass if the embedding is not already generated. 
            last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            embedding = torch.mean(last_hidden_state, 1)

        logits = self.classifier(embedding)
        logits = torch.nn.functional.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss = binary_cross_entropy(torch.reshape(logits, labels.size()), labels.to(logits.dtype))
        
        return logits, loss
        

class EsmClassifierV1(torch.nn.Module):
    
    def __init__(self, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifier object, as well as the torch.nn.Module superclass. 
        '''
        # Initialize the super class, torch.nn.Module. 
        super(EsmClassifier, self).__init__()

        self.model = EsmForSequenceClassification.from_pretrained(name, num_labels=1)

        # Freeze all model weights which aren't related to the classifier. 
        for name, param in self.model.esm.named_parameters():
            param.requires_grad = False
   
    def forward(self, input_ids=None, attention_mask=None, labels=None, index=None, embeddings=None):
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


def esm_train(model, train_dataloader, test_dataloader=None, n_epochs=300):
    '''
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 

    # losses = {'train':[], 'test':[], 'accuracy':[]}
    losses = {'train':[], 'test':[]}

    optimizer = Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train() # Put the model in training mode. 

    for epoch in tqdm(range(n_epochs), desc='Training classifier...'):

        batch_count = 0
        batch_total = len(train_dataloader)

        for batch in train_dataloader:
            print(f'BATCH {batch_count}/{batch_total}\t', end='\r')
            batch_count += 1

            batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 

            logits, loss = model(**batch)
            loss.backward() #
            optimizer.step() 
            optimizer.zero_grad()
        
        losses['train'].append(loss.item()) # Add losses to a history. 
        
        # if test_loader is not None:
        #     # test_loss, accuracy = esm_test(model, test_loader)
        #     test_loss, _ = esm_test(model, test_loader)
        #     losses['test'].append(test_loss.item())
        #     # losses['accuracy'].append(accuracy.item())
        #     # Make sure to put the model back in train mode. 
        #     model.train()

    return losses


# TODO: Fix how labels are managed and stored. 
def esm_test(model, test_loader):
    '''
    Evaluate the model on the test data. 
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 
    
    model.eval() # Put the model in evaluation mode. 

    preds, labels, loss = [], [], []
    for batch in tqdm(test_loader, desc='Calculating batch loss...'): 
        batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 
        
        with torch.no_grad():
            batch_logits, batch_loss = model(**batch)
        loss.append(batch_loss.expand(1))

        preds.append(torch.round(batch_logits)) # .to(device) # To zero or one.
        labels.append(batch.get('labels', None))

    # Concatenate the accumulated results. 
    preds = torch.flatten(torch.cat(preds))

    try: # If labels are given, losses will be calculated. 
        labels = torch.cat(labels)
        loss = torch.cat(loss).mean()
        return {'preds':preds, 'loss':loss.item(), 'labels':labels}
    except:
        return {'preds':preds, 'loss':None, 'labels':None}






