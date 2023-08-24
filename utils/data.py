'''
Utility functions for gathering and preprocessing data.
'''

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import date
import re
import io
import h5py
import sklearn.cluster
from tqdm import tqdm
import random
import time
import subprocess

from plot import *


def get_id(head):
    '''Extract the unique identifier from a FASTA metadata string (the 
    information on the line preceding the actual sequence). This ID should 
    be flanked by '|'.
    '''
    start_idx = head.find('|') + 1
    # Cut off any extra stuff preceding the ID, and locate the remaining |.
    head = head[start_idx:]
    end_idx = head.find('|')
    return head[:end_idx]


def clstr_to_df(clstr):
    '''Convert a .clstr file string to a pandas DataFrame. The resulting 
    DataFrame maps cluster label to gene ID.'''

    df = {'id':[], 'cluster':[]}

    # The start of each new cluster is marked with a line like ">Cluster [num]"
    clusters = re.split(r'^>.*', clstr, flags=re.MULTILINE)
    # Split on the newline. 
    for i, cluster in enumerate(clusters):
        ids = [get_id(x) for x in cluster.split('\n') if x != '']
        df['id'] += ids
        df['cluster'] += [i] * len(ids)

    return pd.DataFrame(df)


def fasta_to_df(fasta):
    '''Convert a FASTA file string to a pandas DataFrame. The 
    resulting DataFrame contains a column for the sequence and a column for 
    the unique identifier.'''

    ids = [get_id(head) for head in re.findall(r'^>.*', fasta, re.MULTILINE)]
    seqs = re.split(r'^>.*', fasta, flags=re.MULTILINE)[1:]
    # Strip all of the newline characters from the amino acid sequences. 
    seqs = [s.replace('\n', '') for s in seqs]
    
    df = pd.DataFrame({'seq':seqs, 'id':ids})
    df = df.astype({'id':'str', 'seq':'str'})
    df = df.set_index('id')

    return df


def df_to_fasta(df, path=None, textwidth=80, truncated=False):
    '''Convert a DataFrame containing FASTA data to a FASTA file format.'''

    fasta = ''
    for row in df.itertuples():
        # If the sequence has been truncated, mark with a special symbol. 
        if truncated:
            fasta += '>|' + row.id + '*|\n'
        else:
            fasta += '>|' + row.id + '|\n'

        # Split the sequence up into shorter, sixty-character strings.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]

        assert ''.join(seq) == row.seq # Make sure no information was lost. 

        seq = '\n'.join(seq) + '\n'
        fasta += seq
    
    # Write the FASTA string to the path-specified file. 
    write(fasta, path=path)

    return fasta


def truncate_sec(fasta, path=None):
    '''Truncate the selenoproteins stored in the inputted file at the first 
    selenocysteine residue. Overwrites the original file with the truncated 
    sequence data.'''
    df = fasta_to_df(fasta)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():
        
        # Find indices where a selenocysteine occurs. 
        idxs = np.where(np.array(list(row.seq)) == 'U')[0]
        # Sequentially truncate at each selenocysteine redidue. 
        seqs = [row.seq[:idx] for idx in idxs]
        # Add new truncated sequences, and the corresponding gene IDs, to the
        # dictionary which will become the new DataFrame. 
        df_trunc['id'] += [row.Index] * len(seqs)
        df_trunc['seq'] += seqs

    df_trunc = pd.DataFrame(df_trunc)
    # Make sure to mark as truncated. 
    fasta = df_to_fasta(df_trunc, path=path, truncated=True)

    return fasta # Return a FASTA file with the truncated proteins. 


def h5_to_csv(path, batch_size=100):
    '''Convert a data file in HD5 format to a CSV file, 
    so that it is easier to work with.'''

    file_ = h5py.File(path)
    keys = list(file_.keys())

    batches = [keys[i:min(i + batch_size, len(keys))] for i in range(0, len(keys), batch_size)]
    assert len(batches) == len(keys) // batch_size + 1

    header = True

    for batch in tqdm(batches):
        data = np.array([file_[k][()] for k in batch])
        df = pd.DataFrame(data)
        df['id'] = [get_id(k) for k in batch]

        # df = df.set_index('id')
        df.to_csv(path.split('.')[0] + '.csv', mode='a', header=header)

        # After the first loop, don't write the headers to the file. 
        header = False

    file_.close()


def sample_kmeans_clusters(kmeans, n):
    '''Sample n elements such that the elements are spread across a set
    of K-means clusters of. Returns the indices of data elements in the sample '''
    
    # Sample from each cluster. 
    n_clusters = kmeans.cluster_centers_.shape[0]
    n = n // n_clusters # How much to sample from each cluster. 
    
    idxs = []
    for cluster in range(n_clusters):
        # Get indices for all data in a particular cluster, and randomly select n. 
        cluster_idxs = np.where(kmeans.labels_ == cluster)[0]
        idxs += list(np.random.choice(cluster_idxs, min(n, len(cluster_idxs))))

    idxs = np.unique(idxs)

    return np.array(idxs)


def filter_sprot(data, n=20000, n_clusters=500):
    '''Filters the SwissProt database by selecting representative proteins. The approach for doing this
    is based on K-means clustering the embedding data, and sampling from each cluster.

    args:
        - data (pd.DataFrame): DataFrame containing the embedding data. 
        - n (int): The approximate number of items which should pass the filter.
        - n_clusters (int): The number of clusters for KMeans.
        - path (str): If specified, the path where the filtering data should be written. 

    
    '''
    # kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(data.values)

    idxs = sample_kmeans_clusters(kmeans, n)

    # Return the filter indices, as well as the cluster labels.
    return idxs, kmeans.labels_


# Maybe need to evaluate clustering?



# def create_feature_vector(seq=None, domain=None):
#     '''Convert an amino acid sequence into a numerical vector containing
#     length, domain information, and amino acid composition'''

#     aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
#     aa_to_int = {aas[i]: i for i in range(len(aas))}
#     domain_to_int = {'prokaryote':0, 'eukaryote':1, 'archea':2}
    
#     # Map each amino acid to an integer. 
#     seq = np.array([aa_to_int_map[aa] for aa in seq])

#     # Create an array of zeroes with the number of map elements as columns. 
#     # This will be the newly-embeddings sequence. 
#     vec = np.zeros(shape=(len(seq), len(aa_to_int)))
#     vec[np.arange(len(seq)), seq] = 1
#     # Also need to normalize according to sequence length. 
#     vec = np.sum(vec, axis=0) / len(seq)

#     # Add domain information to the array. 
#     vec = np.concatenate([np.zeros(3), vec], axis=1)
#     vec[domain_to_int[domain]] = 1

#     # Add a length feature to the array. 
#     vec = np.concatenate([np.array([len(seq)]), vec], axis=1)

#     return vec


def train_test_split(data, clusters=None, train_size=None):
    '''Split the sequence data into a training and test set.
    
    args:
        - data (pd.DataFrame)
        - clusters (pd.DataFrame)
    '''
    assert clusters is not None
    assert data.index.name == 'id'

    # Going to have issues with things being 


    test_size = len(data) - train_size

 

def write(text, path):
    '''Writes a string of text to a file.'''
    if path is not None:
        with open(path, 'w') as f:
            f.write(text)


def read(path):
    '''Reads the information contained in a text file into a string.'''
    with open(path, 'r') as f:
        text = f.read()
    return text


def get_sec_metadata_from_uniprot(fields=['lineage'], path=None):
    '''Download metadata from UniProt in TSV format using their API. By default,
    downloads the taxonomic lineag
e, as well as the eggNOG orthology group.'''

    fields = '%2C'.join(fields)
    url = f'https://rest.uniprot.org/uniprotkb/stream?compressed=false&fields=accession%2C{fields}&format=tsv&query=%28%28ft_non_std%3Aselenocysteine%29%29'
    tsv = requests.get(url).text

    if 'lineage' in fields:
        # Clean up the metadata by removing all non-domain taxonomic information.
        func = lambda l : l.split(',')[1].strip().split(' ')[0].lower()
        df = pd.read_csv(io.StringIO(tsv), delimiter='\t')
        df['domain'] = df['Taxonomic lineage'].apply(func)
        df = df.drop(columns=['Taxonomic lineage'])
        # df = df.rename(columns={'Entry':'id', 'eggNOG':'cog'})
        df = df.set_index('id')

        # Convert back into a text string. 
        tsv = df.to_csv(sep='\t')
        write(tsv, path=path)

    return tsv


def get_sec_from_df(df):
    pass


def get_sec_from_uniprot(path=None):
    '''Uses the UniProt API to download all known selenoproteins from the database.
    It returns a text string with the information in FASTA format.'''

    url = 'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28ft_non_std%3Aselenocysteine%29%29'
    fasta = requests.get(url).text

    write(fasta, path)

    return fasta


def get_homology_clusters(path, c=0.8, o=None):
    ''''''

    filename = path.split('/')[-1].split('.')[0]
    data_dir = '/'.join(path.split('/')[:-1]) + '/'
    cmd = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'

    # Run the CD-HIT command on data stored in the input path. 
    subprocess.run(f'{cmd} -i {path} -o {data_dir}out -c {c}', shell=True)
    # Change the names of the representative sequence file and clstr file
    # to match my naming convention. 
    subprocess.run(f'mv {data_dir}out {data_dir}{filename}_reps.fasta', shell=True)
    subprocess.run(f'mv {data_dir}out.clstr {data_dir}{filename}.clstr', shell=True)

    # Read in the clstr file. 
    df = clstr_to_df(read(f'{data_dir}{filename}.clstr'))




def main(
    data_dir='/home/prichter/Documents/selenobot/data/', 
    figure_dir='/home/prichter/Documents/selenobot/figures/'):

    # today = date.today().strftime('%m%d%y')
    # fasta = get_sec_from_uniprot(path=data_dir + f'uniprot_{today}_sec.fasta'
    # fasta = read(data_dir + 'uniprot_081123_sec.fasta')
    # truncate_sec(fasta, path=data_dir + 'uniprot_081123_sec_trunc.fasta')

    # h5_to_csv(data_dir + 'uniprot_081623_sprot_embeddings.h5')

    get_homology_clusters(data_dir + 'all.fasta')

    # idxs = np.ravel(pd.read_csv(data_dir + 'uniprot_081623_sprot_embeddings_filtered.csv', usecols=['idx']).values)
    # clusters = np.ravel(pd.read_csv(data_dir + 'uniprot_081623_sprot_embeddings_filtered.csv', usecols=['cluster']).values)
    # print('here')

    # Read in the SwissProt data and apply K-means clustering. 
    # data = pd.read_csv(data_dir + 'uniprot_081623_sprot_embeddings_filtered.csv', index_col='id')
    # idxs, clusters = filter_sprot(data, n_clusters=500)
    

    # Read in the gene IDs of the genes which passed the SwissProt filter. 

    # sizes = np.array([np.sum(clusters == x).item() for x in range(500)])
    # plot_distribution(sizes, path=figure_dir + 'sprot_embeddings_cluster_size_distribution_nclusters=500.png', title='Distrubution of K-means cluster sizes\non SwissProt embeddings', xlabel='size', bins=100)

    # Write the filtered data to a file. Used 500 clusters here. 
    # data.iloc[idxs].to_csv(data_dir + 'uniprot_081623_sprot_embeddings_filtered.csv')
    
    # plot_distribution(data.values, path=figure_dir + 'sprot_embeddings_distribution.png', title='Distrubution of PCA-reduced SwissProt embeddings', xlabel='PCA 1')
    # plot_distributions((data.values, data.iloc[idxs].values), path=figure_dir + 'sprot_embeddings_with_filter_distribution_nclusters=1.png', title='Distrubution of PCA-reduced SwissProt embeddings', xlabel='PCA 1')
    
    # plot_filter_sprot(data, idxs=idxs, clusters=clusters, path=figure_dir + 'sprot_filter_scatterplot.png')
    # plot_filter_sprot(data, clusters=clusters, path=figure_dir + 'sprot_filter_cluster_scatterplot.png')

    # # Apply the filter to the data. 
    # data = data.iloc[filter_]
    # data.to_csv(data_dir + 'uniprot_081623_sprot_embeddings_filtered.csv')

if __name__ == '__main__':
    main()

