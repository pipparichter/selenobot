'''
This file contains the definition of a class which uses the pre-trained ProtGPT2 model to 
predict whether or not a protein is a selenoprotein or not. 
'''

import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from dataset import SequenceDataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import pickle

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO Might be worth outputting a list of possible sequences?
# TODO: Need to figure out how a stop codon is represented

# It could be that simply appending a newline character to the end of every sequence will prevent
# GPT from predicting that one next. Not sure what weirdness this might cause though, probably
# better safe than sorry. 

class ProtGPTClassifierV1(torch.nn.Module):
    '''
    First attempt at building a sequence classifier using ProtGPT. Assumes that
    '''
    def __init__(self, name='nferruz/ProtGPT2'):

        super(ProtGPTClassifierV1, self).__init__()

        self.tokenizer = tokenizer
        self.gpt = GPT2LMHeadModel.from_pretrained(name)

    # def _filter_newlines(idxs):
    #     '''
    #     Because newline characters were not removed before training the model,
    #     GPT will fill in newlines for any long sequences which don't have them.
    #     This may interfere with next-token prediction (a newline might be predicted)
    #     right before a STOP -- I have no way of knowing if this will actually happen,
    #     but this accounts for it. 
    #     '''
    #     most_likely_idxs = []
    #     print(idxs)
    #     for i in idxs: # Scanning through the logits for each sequence. 
    #         if i[0].item() == 199:
    #             most_likely_idxs.append(i[1].item())
    #         else: # If the most likely detected character is a newline, skip over it!
    #             most_likely_idxs.append(i[0].item())
        
    #     return torch.Tensor(most_likely_idxs)

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
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


# TODO: Fix how labels are managed and stored. 
def protgpt_test(model, test_dataloader):
    '''
    Evaluate the model on the test data. 
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 
    
    model.eval() # Put the model in evaluation mode. 

    preds, labels = [], []
    accs = []
    pbar = tqdm(test_dataloader, desc=f'Evaluating model on test data | ACCURACY: N/A |', position=0)
    for batch in pbar: 
        batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 
        
        with torch.no_grad():
            preds.append(model(**batch))
        labels.append(batch.get('labels', None))
        
        # Update the progress bar. 
        if labels[-1] is not None:
            # Calculate the batch accuracy and add it to the list. 
            accs.append((preds[-1] == labels[-1]).float().mean().item() * 100)
            # print(accs)
            pbar.set_description(f'Evaluating model on test data | ACCURACY: {int(np.mean(accs))} % |')
    pbar.close()
    
    # Concatenate the accumulated results. 
    preds = torch.flatten(torch.cat(preds))
    labels = None if (labels[0] is None) else torch.cat(labels)
    return {'preds':preds, 'labels':labels}


if __name__ == '__main__':
    train_data = pd.read_csv('/home/prichter/Documents/protex/data/train.csv')
    test_data = pd.read_csv('/home/prichter/Documents/protex/data/test.csv')
    
    # NOTE: GPT2 Tokenizer seems to be way slower than the AutoTokenizer... 
    # tokenizer = GPT2Tokenizer.from_pretrained('nferruz/ProtGPT2')
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
    tokenizer.pad_token = '[PAD]' # Doesn't seem to be using the padding token for padding...

    kwargs = {'tokenizer':tokenizer, 'return_tensors':'pt', 'padding':True}
    test_dataset = SequenceDataset(test_data, labels=test_data['label'].values, **kwargs)
    train_dataset = SequenceDataset(train_data, labels=train_data['label'].values, **kwargs)

    model = ProtGPTClassifierV1(tokenizer=tokenizer)
    
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)

    result = protgpt_test(model, test_dataloader)
    result = pd.DataFrame(result)

    # result.to_csv('./protgpt_test_predictions.csv')





