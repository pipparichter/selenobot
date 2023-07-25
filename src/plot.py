'''
Plotting utilities for the ProTex tool. 
'''
import matplotlib.pyplot as plt
import seaborn as sns
import dataset 
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA

colors = sns.color_palette('Paired')
palette = 'Paired'


def plot_embeddings(embeddings, n_points=500, labels=None, filename='embeddings_esm.png', title='ESM-generated protein embeddings'):
    '''
    Apply PCA dimensionality reduction to the tokenized amino acid sequences (both those generated
    simply by counting amino acid content, and those generated using ESM). 

    args:
        - embeddings
        - n_points
        - labels
    '''
    # Get rid of the index values prior to PCA application. 
    embeddings = embeddings.drop(columns=['index']).values
    # Sample rows from the array without replacement. Also get rid of index. 
    sample_idxs =  np.random.choice(len(embeddings), size=n_points, replace=False)
    embeddings = embeddings[sample_idxs, 1:]
    # print(embeddings[:, 0]) # Still picking up the index. 
    
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    
    # Instantiate the PCA model for dimensionality reduction.  
    # pca = PCA(n_components=2)
    umap = UMAP(n_components=2) # UMAP seems to work a bit better than PCA. 
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # fit_transform should take in data of (n_samples, n_features)
    # data = pd.DataFrame(pca.fit_transform(embeddings), columns=['PCA 1', 'PCA 2'])
    data = pd.DataFrame(umap.fit_transform(embeddings), columns=['UMAP 1', 'UMAP 2'])

    if labels is not None:
        data['label'] = labels[sample_idxs]
        sns.scatterplot(data=data, x='UMAP 1', y='UMAP 2', ax=ax, hue='label', palette=palette)
        # sns.scatterplot(data=data, x='PCA 1', y='PCA 2', ax=ax, hue='label', palette=palette)
        # ax.legend(['short', 'selenoprotein'])
    else:
        sns.scatterplot(data=data, x='UMAP 1', y='UMAP 2', ax=ax)

    fig.savefig(filename, format='png')


def plot_dataset_length_distributions(sec_data, short_data):
    '''
    Visualize the length distributions of the short and truncated selenoproteins. 

    args:
        - sec_data (pd.DataFrame)
        - short_data (pd.DataFrame)
    '''
    # sec_data = dataset.fasta_to_df('/home/prichter/Documents/protex/data/sec.fasta')
    # short_data = dataset.fasta_to_df('/home/prichter/Documents/protex/data/sec.fasta')

    plot_data = {}

    # Grab the sequence lengths. 
    sec_lengths = sec_data['seq'].apply(len).to_numpy()
    short_lengths = short_data['seq'].apply(len).to_numpy()
    lengths = np.concatenate([sec_lengths, short_lengths])

    plot_data['length'] = lengths
    plot_data['dataset'] = np.array(['sec'] * len(sec_lengths) + ['short'] * len(short_lengths))
    plot_data = pd.DataFrame(plot_data) # Convert to DataFrame. 

    fig, ax = plt.subplots(1)

    # Add some relevant information to the plot. 
    ax.text(700, 1000, f'sec.fasta size: {len(sec_lengths)}\nshort.fasta size: {len(short_lengths)}', size='small', weight='semibold')
    ax.set_title('Length distributions over sequence datasets')

    sns.histplot(data=plot_data, hue='dataset', x='length', legend=True, ax=ax, palette=palette, multiple='dodge', bins=15)

    fig.savefig('dataset_length_distributions.png', format='png')


def plot_train_test_composition(train_data, test_data, train_labels, test_labels):
    '''
    '''

    # Label is 1 if the sequence is a selenoprotein. 
    train_sec_count = sum(train_labels)
    test_sec_count = sum(test_labels)

    labels = ['sec', 'short']

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].pie([train_sec_count, len(train_data) - train_sec_count], labels=labels, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Training set composition')

    axes[1].pie([test_sec_count, len(test_data) - test_sec_count], labels=labels, autopct='%1.1f%%', colors=colors)
    axes[1].set_title('Testing set composition')

    fig.savefig('train_test_composition.png', format='png')
   


