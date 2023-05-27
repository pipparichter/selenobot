'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''

import transformers # Import the pretrained models from HuggingFace.
from transformers import EsmForSequenceClassification, EsmModel, AutoTokenizer
from main import fasta_to_df, embedding_to_txt, generate_labels
import torch

# NOTE: ESM has a thing which, by default, adds a linear layer for classification. Might be worth 
# training my own version of this layer in the future. 

def ESM():
    '''
    Supports three approaches to classifying sequences using the ESM model: (1) the built-in EsmForSequenceClassification and
    (2) using the entire embedding, not just the pool layer, to classify the sequences. 
    '''
    def __init__(self, labels):
        '''

        kwargs:
            : labels (list): For now, only a single-label classification problem is being supported. 
        '''
        self.name = 'facebook/esm2_t6_8M_UR50D'      
        self.labels = torch.tensor(labels) # Convert specified labels to a tensor. 

        # I think I need a better handle on what the tokenizer is doing... 
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        
        # This is a single-label classification problem -- a protein is either truncated or full-length. 
        self.classification_model = EsmForSequenceClassification.from_pretrained(self.name, num_labels=len(labels))
        # TODO: Possibly train my own, final layer on only microbial protein data, maybe. 
        self.standard_model = EsmModel.from_pretrained(self.name)


    def classify(self, read_from='test.fasta'):
        '''
        '''
        data = list(fasta_to_df(read_from=read_from)['seq'])
        inputs = self.tokenizer(data)

        self.pretrained_classification_model(**inputs, labels=self.labels)


    def embed(self, read_from='sec_trunc.fasta', write_to='sec_trunc.txt'):
        ''' 
        Generate embeddings using a pretrained model. Writes all embeddings to a file. 
        '''
        # Get the sequence data from the DataFrame. Needs to be in the form of a python list
        data = list(main.fasta_to_df(read_from=read_from)['seq'])

        chunk_size = 10
        n = len(seq_data) // chunk_size
        
        for i in tqdm(range(n), 'Processing chunks...'):
            # Data should be a list of strings. 
            chunk = data[chunk_size * i : chunk_size * i + chunk_size]

            chunk = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, is_split_into_words=False, max_length=1024)
            # NOTE: Should I be using the pooling layer here?
            embedding = model(**chunk).last_hidden_state
            # Had a shape of torch.Size([batch_size, sequence_length, embedding_size]). 
            # For now, handle this output by taking the average over the embeddings for each amino acid position.
            embedding = tf.mean(embedding, dim=1)
            # Need to write to file as we go. 
            embedding_to_txt(embedding)

    # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important 
    def classify(self, use_builtin=True):
        '''

        kwargs:
            : use_builtin (bool): True by default. Whether or not to use the built-in ESM classifier layer. 
        '''

        pass


if __name__ == '__main__':
    esm = ESM()

