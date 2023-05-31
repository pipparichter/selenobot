'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''

import transformers # Import the pretrained models from HuggingFace.
from transformers import EsmForSequenceClassification, EsmModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AdamW
from torch.utils.data import DataLoader

# NOTE: ESM has a thing which, by default, adds a linear layer for classification. Might be worth 
# training my own version of this layer in the future. 


class ESMClassifier(torch.nn.Module):
    
    def __init__(self, use_builtin_classifier=True, freeze_pretrained=True, num_labels=1, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifier object, as well as the torch.nn.Module superclass. 
        
        kwargs:


        '''
        # Initialize the super class, torch.nn.Module. 
        super(ESMClassifier, self).__init__()

        self.use_builtin_classifier = use_builtin_classifier # Whether or not the buil-in sequence classifier is used. . 
        self.loss_func = torch.nn.CrossEntropyLoss()

        # NOTE: Do I need to freeze ESM model weights when I do this?
        if use_builtin_classifier:
            self.esm_classifier = EsmForSequenceClassification.from_pretrained(name, num_labels=num_labels)
        else: # Need an additional classifier layer if we are going to be fine-tuning. 
            self.esm = EsmModel.from_pretrained(name)
            # Eventually, make the classifier more complex?
            self.classifier = torch.nn.Linear(self.esm.config.hidden_size, num_labels)

        if freeze_pretrained:
            if use_builtin_classifier:
                for param in self.esm_classifier.esm.parameters():
                    param.requires_grad = False
            else:
                for param in self.esm.parameters():
                    # Make it so that none of the pretrained weights can be updated. 
                    param.requires_grad = False
    

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
        if self.use_builtin_classifier:
            # NOTE: I think loss is already calculated here?
            return self.esm_classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # If the regular ESM model is used, output type is BaseModelOutputWithPoolingAndCrossAttentions
        else:
            # If the regular ESM moel is used, then it doesn't take labels as an input. 
            esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            # Pooler output has a shape of (batch_size, hidden_size)
            pooler_output = esm_output.pooler_output
    
            logits = self.classifier(pooler_output)
            loss = None
            if labels is not None:
                loss = self.loss_func(torch.flatten(logits), torch.flatten(labels))

            # Should I be returning the hidden_states of the classifier layer here?
            # return SequenceClassifierOutput(loss=loss, logits=logits, atttentions=esm_output.attentions, hidden_states=esm_output.hidden_states)
            return SequenceClassifierOutput(loss=loss, logits=logits, attentions=esm_output.attentions, hidden_states=esm_output.hidden_states)


def esm_train(model, train_data, batch_size=10, shuffle=True, n_epochs=100):
    '''
    '''
    losses = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters())

    model.train() # Put the model in training mode. 

    # pbar = tqdm(total=n_epochs * batch_size, desc='Processing batches...')

    for epoch in tqdm(range(n_epochs), desc='Training classifier...'):
        
        for batch in train_loader:
            # pbar.update(1)

            # optimizer.zero_grad() 
            # print(batch['input_ids'], batch['input_ids'].size())
            # print(batch['labels'], batch['labels'].size())

            outputs = model(**batch)

            loss = outputs.loss
            loss.backward() #

            optimizer.step() # What does this do?
            optimizer.zero_grad()
        
        losses.append(loss.item()) # Add losses to a history. 

    # pbar.close()

    # Do I need to return the model here, or is everything adjusted inplace?
    return losses

# NOTE: What is all the stuff with cuda? It seeme like it's a way to mount things on a GPU,
# although not sure if my machine has this. 

def esm_test(model, test_data, shuffle=True, batch_size=10):
    '''
    '''

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    model.eval() # Put the model in evaluation mode. 

    for batch in eval_dataloader:
        # Does this just insure that the gradients are not being computed?
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits # Get the output of the classification layer. 
        # The logits should be a one-dimensional vector
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar_eval.update(1)
        
    print(metric.compute())

if __name__ == '__main__':
    pass