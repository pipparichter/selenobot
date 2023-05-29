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
    
    def __init__(self, use_pretrained=True, num_labels=1, name='facebook/esm2_t6_8M_UR50D', loss_func=torch.nn.CrossEntropyLoss):
        '''
        Initializes an ESMClassifier object, as well as the torch.nn.Module superclass. 
        
        kwargs:
            - use_pretrained (bool): Whether or not to use the pretrained ESM sequence classifier. 
            - num_labels (int): Number of classes. This will be 1 for the forseeable future.
            - name (str): Name of the pretrained ESM model to load.  
            - loss_func (): The loss function to use for fine-tuning. 

        '''
        # Initialize the super class, torch.nn.Module. 
        super(ESMClassifier, self).__init__()

        self.num_labels = 1 # Currently only one class. 
        self.pretrained = use_pretrained # Whether or not the pretrained model is being used. 
        self.loss_func = loss_func

        # I think I need a better handle on what the tokenizer is doing... 
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        
        # NOTE: Do I need to freeze ESM model weights when I do this?
        if use_pretrained:
            self.esm = EsmForSequenceClassification.from_pretrained(name)
        else: # Need an additional classifier layer if we are going to be fine-tuning. 
            self.esm = EsmModel.from_pretrained(name, num_labels=self.num_labels)
            self.classifier = torch.nn.Linear(self.esm.config.hidden_size, self.num_labels)
    

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
        output = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        if self.pretrained:
            return output # This is already the correct return type.  
        # If the regular ESM model is used, output type is BaseModelOutputWithPoolingAndCrossAttentions
        else:
            output = output.pooler_output # Shape of (batch_size, hidden_size)
            logits = self.classifier(output)

            loss = None # Calculate the loss. 
            if labels is not None:
                loss = self.loss_func(logits, labels)

            return SequenceClassifierOutput(loss=loss, logits=logits, atttentions=output.attentions, hidden_states=output.hidden_states)


def esm_train(model, train_data, batch_size=1, shuffle=True, n_epochs=100):
    '''
    '''
    losses = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters())

    model.train() # Put the model in training mode. 

    for epoch in tqdm(range(n_epochs)):

        for batch in train_loader:

            optimizer.zero_grad() # What does this do?
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            loss.backward() # What does this do?

            optimizer.step() # What does this do?
            optimizer.zero_grad()



def esm_test(model, test_data):

        model.eval() # Put the model in evaluation mode. 

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)
            
        print(metric.compute())

if __name__ == '__main__':
    pass