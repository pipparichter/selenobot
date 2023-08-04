'''
A generic linear classifier which sorts embedded amino acid sequences into two categories:
selenoprotein or non-selenoprotein. 
'''
import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.nn.functional import cross_entropy, binary_cross_entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# What functionality will span all classifier objects? Possibly put the 
# test and train functions here?
class Classifier(torch.nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()
        

class NextTokenClassifier(Classifier):
    '''
    '''
    def __init__(self):

        super(NextTokenClassifierV1, self).__init__()
        self.gpt = transformers.GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2')

    def forward(self, **kwargs):
        return self._v1(**kwargs)

    def _v3(self, input=None, labels=None):
        pass
    
    def _v2(self, input=None, labels=None):
        pass

    def _v1(self, input=None, labels=None):
        '''
        '''
        # NOTE: Not sure if looping like this makes it way more inefficient. 

        preds = []
        # for id_, mask in zip(input_ids, attention_mask):
        for id_, mask, label in zip(input_ids, attention_mask, labels):
            id_ = id_[mask.to(torch.bool)]
            logits = self.gpt(id_).logits[-1] # Grab the logit for the last sequence element. 

            idxs = torch.argsort(logits, descending=True).numpy()
            # The 199 index represents the newline character, which should be discarded. 
            # See https://huggingface.co/nferruz/ProtGPT2/discussions/20. 
            next_token_idx = idxs[0] if (idxs[0] != 199) else idxs[1]

            print(label, self.tokenizer.decode(next_token_idx))

            if next_token_idx == 0: # This is the index for the end-of-text character. 
                preds.append(0)
            else:
                preds.append(1)

        # Putting the entire batch through the model at once was taking over 30 seconds per 10-element
        # batch, and was making my laptop tweak out. Accuracy was also weirdly low. Feeding in sequences
        # without all the padding, one-by-one, seems to be much faster. Not sure why.  

        # # Logits will have dims (batch_size, sequence_length, vocab_size)
        # # The -1 indexing grabs the last element of the sequence. 
        # logits = self.gpt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        # idxs = torch.argsort(logits, dim=-1, descending=True)
        # idxs = ProtGPTClassifier._filter_newlines(idxs) # Remove the newline characters from consideration. 
        # # Seems as though the EOS character is indicated by zero. Newline is \n, which should be ignored. 
        # preds = torch.Tensor([0 if (i == 0) else 1 for i in idxs])
        # print(preds)

        return torch.Tensor(preds)



class EmbeddingClassifier(Classifier):
    '''
    '''
    
    def __init__(self):

        # Initialize the super class, torch.nn.Module. 
        super(EmbeddingClassifier, self).__init__()


        # Should be 320... 
        # hidden_size = self.esm.get_submodule('encoder.layer.5.output.dense').out_features
        self.latent_dim = None
        self.classifier = torch.nn.Linear(self.self.latent_dim, 1)


    def forward(self, embedding=None, labels=None):
        '''
        A forward pass of the EmbeddingClassifier
        '''
        logits = self.classifier(embedding)
        logits = torch.nn.functional.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss = binary_cross_entropy(torch.reshape(logits, labels.size()), labels.to(logits.dtype))
        
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

 