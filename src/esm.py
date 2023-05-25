'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''

import transformers # Import the pretrained models from HuggingFace.

# Not totally sure what each part of this model name means
# model_name = 'facebook/esm2_t6_8M_UR50D'
# model_name = 'facebook/esm1b_t33_650M_UR50S'

def embed(model, tokenizer, read_from='sec_trunc.fasta', write_to='sec_trunc.txt'):
    '''
    Generate embeddings using a pretrained model. Writes all embeddings to a file. 
    '''
    # Get the sequence data from the DataFrame. Needs to be in the form of a python list
    seq_data = list(fasta_to_df(read_from=read_from)['seq'])

    chunk_size = 10
    n = len(seq_data) // chunk_size
    
    with open(write_to, 'w', encoding='utf8') as f: 
        for i in tqdm(range(n), 'Processing chunks...'):
            # Data should be a list of strings. 
            chunk = seq_data[chunk_size * i : chunk_size * i + chunk_size]

            chunk = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, is_split_into_words=False, max_length=1024)
            embedding = model(**chunk).last_hidden_state
            # Had a shape of torch.Size([batch_size, sequence_length, embedding_size]). 
            # For now, handle this output by taking the average over the embeddings for each amino acid position.
            embedding = tf.mean(embedding, dim=1)
            # Need to write to file as we go. 
            np.savetxt(f, embedding.detach().numpy(), delimiter=' ', newline='\n')

