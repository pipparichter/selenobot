'''A script for summarizing information about the resulting Datasets.'''

import pandas as pd
import os
import numpy as np
import scipy.stats
from utils import fasta_ids_with_min_seq_length, fasta_ids, fasta_seqs, csv_ids, pd_from_fasta_with_min_seq_length, pd_from_clstr

DATA_DIR = '/home/prichter/Documents/selenobot/data/'
GTDB_DATA_DIR = '/home/prichter/Documents/selenobot/data/gtdb/'
UNIPROT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot/'

DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/detect/'
MIN_SEQ_LENGTH = 6 # From the setup.py file. 

# https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Creviewed%2Cid%2Cgene_names%2Corganism_name%2Cdate_created%2Cversion&format=tsv&query=%28*A0A1B0GTW7%29

# Questions
# 1. What is methionine content in truncated versus non-truncated proteins? Josh has a theory that this
#   is why the AacEmbedder-based classifier is working so well, because methionine has a higher ratio.
#   I am a little skeptical, because I suspect this is just the same as length. 
# 2. Why are the results so different with different selenoprotein_fractions?

# Keeps failing tests related to proteins which are present in all_data but not in the partitioned dataset. 
# Want to figure out which proteins are present in all_data which are not present in training, testing, and validation datasets. 
def summary_missing_proteins():
    '''Characterize proteins which are missing in the training, testing, and validation datasets.'''
    f = 'summary.summary_missing_proteins'
    ids = np.concatenate([csv_ids(os.path.join(DETECT_DATA_DIR, file)) for file in ['train.csv', 'test.csv', 'val.csv']]).ravel()
    all_data = pd_from_fasta_with_min_seq_length(os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))

    missing_ids = set(all_data['id']) - set(ids)
    assert len(missing_ids) > 0, f'{f}: No missing proteins found.'

    all_data = all_data[np.isin(all_data['id'].values, list(missing_ids), assume_unique=True)]
    all_data['length'] = all_data['seq'].apply(len)

    # Maybe it's related to the cluster number?
    clstr_data = pd_from_clstr(os.path.join(UNIPROT_DATA_DIR, 'all_data.clstr'))
    all_data['cluster'] = [clstr_data['cluster'][clstr_data['id'] == id_].item() for id_ in all_data['id']]

    # They might be being lost in the merge in setup.add_embeddings_to_file.
    ids_with_embeddings = csv_ids(os.path.join(UNIPROT_DATA_DIR, 'all_embeddings.csv'))
    all_data['has_embedding'] = [id_ in ids_with_embeddings for id_ in all_data['id']]

    # Might have to do with the date entries were added to UniProt. Should check this next. 

    print('missing proteins ', '-' * (100 - len('missing proteins ')), '\n')
    print(all_data.set_index('id'))
    print(f'{f}:', np.sum(all_data['has_embedding'].values), 'of the missing proteins have corresponding PLM embeddings in the all_embeddings.csv file.')


def summary(vars=['met_content', 'length']):
    seqs = utils.fasta_seqs(os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))
    ids = utils.fasta_ids(os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))

    data = {'length':[], 'met_content':[], 'label':[]}
    for seq, id_ in zip(seqs, ids):
        if len(seq) >= MIN_SEQ_LENGTH:
            data['length'].append(len(seq))
            data['met_content'].append(seq.count('M') / len(seq))
            data['label'].append(1 if '[1]' in id_ else 0)
    data = pd.DataFrame(data)
    # Compare the distributions for truncated versus not truncated.
    # I am curious about how the batch sampler alters this score. 

    for var in vars:
        print(var, '-' * (50 - len(var)), '\n')
        p = scipy.stats.mannwhitneyu(np.ravel(data[var][data['label'] == 1]), np.ravel(data[var][data['label'] == 0])).pvalue.item()
        print(f'Mann-Whitney U p-value: {p}\n')

        var_data = data[['label', var]].groupby('label').aggregate(['mean', 'median', 'std', 'max', 'min']).droplevel(0, axis=1)
        # data['length']['new'] = data['length']['mean'].astype(int)
        print(var_data)
        print()


if __name__ == '__main__':
    # summary()
    summary_missing_proteins()


# def main(log_file_path=None):
#     '''Print the results of the setup procedure.'''
    
#     if log_file_path is not None: # Write the summary to a log file if specified. 
#         with open(log_file_path, 'w', encoding='UTF-8') as f:
#             sys.stdout = f

#     for file in os.listdir(DETECT_DATA_DIR):
#         path = os.path.join(DETECT_DATA_DIR, file)
#         print(f'[{file}]')
#         print('size:', csv_size(path))
#         sec_content = pd.read_csv(path, usecols=['label'])['label'].values.mean()
#         print('selenoprotein content:', np.round(sec_content, 3))
#         print()

#     print('[all_data.fasta]')
#     path = os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta')
#     seqs = fasta_seqs(path)
#     ids = fasta_ids(path)
#     print('total sequences:', len(seqs))
#     print('total selenoproteins:', len([i for i in ids if '[' in i]))
#     print(f'sequences of length >= {MIN_SEQ_LENGTH}:', np.sum(np.array([len(s) for s in seqs]) >= MIN_SEQ_LENGTH))
#     print(f'selenoproteins of length >= {MIN_SEQ_LENGTH - 1}:', len([i for i, s in zip(ids, seqs) if '[' in i and len(s) >= MIN_SEQ_LENGTH]))
#     print()

#     print('[all_data.fasta]')
#     path = os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta')
#     seqs = fasta_seqs(path)
#     ids = fasta_ids(path)
#     print('total sequences:', len(seqs))
#     print('total selenoproteins:', len([i for i in ids if '[' in i]))
#     print(f'sequences of length >= {MIN_SEQ_LENGTH}:', np.sum(np.array([len(s) for s in seqs]) >= MIN_SEQ_LENGTH))
#     print(f'selenoproteins of length >= {MIN_SEQ_LENGTH - 1}:', len([i for i, s in zip(ids, seqs) if '[' in i and len(s) >= MIN_SEQ_LENGTH]))
#     print()
