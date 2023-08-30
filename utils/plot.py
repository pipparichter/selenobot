'''
Plotting utilities for the ProTex tool. 
'''
import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics import confusion_matrix
import sklearn.decomposition
import matplotlib as mpl


palette = 'Set2'


# def plot_distribution(data, ax=None, path=None, title=None, xlabel=None, **kwargs):
#     '''Visualize the distribution of a set of data. 


#     '''
#     # assert type(data) == np.array, 'Input data must be a numpy array.'
#     assert ax is not None

#     sns.histplot(data=data, ax=ax, legend=False,  **kwargs)

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)

#     # If the data passed into the function is more than one-dimensional, 
#     # apply dimensionality reduction to one dimension. 
#     if len(data.shape) > 1 and (data.shape[-1] > 1):
#         pca = sklearn.decomposition.PCA(n_components=1)
#         data = pca.fit_transform(data.values)

#     # Data should be one-dimensional following PCA reduction. 
#     data = np.ravel(data)

#     if path is not None:   
#         try:
#             fig.savefig(path, format='png')
#         except NameError:
#             print('Figure could not be saved.')


# def plot_distributions(data, title=None, path=None, xlabel=None, ax=None, **kwargs):
#     '''Plots multiple distributions on the same histogram by calling the
#     plot_distribution function on each dataset specified.
    
#     args:
#         - data (tuple): A tuple where each element is a dataset whose
#             distribution to plot. 
#     ''' 
#     assert type(data) == tuple
#     assert ax is not None
#     assert len(data) > 1 # Make sure more than one dataset is speciied.
#     assert np.all([d.shape[-1] == data[0].shape[-1] for d in data])

#     # We will want to reduce each of the datasets in the same PCA pass. 
    
#     # Get the sizes of each dataset passed in to the function. 
#     # This will allow us to extract individual datasets after combination. 
#     sizes = [len(d) for d in data]
#     data = np.concatenate(data, axis=0)
#     # If data is multi-dimensional, reduce using PCA. 
#     if len(data.shape) > 1 and (data.shape[-1] > 1):
#         pca = sklearn.decomposition.PCA(n_components=1)
#         data = pca.fit_transform(data)

#     # Data should be one-dimensional following PCA reduction. 
#     data = np.ravel(data)

#     fig, ax = plt.subplots(1)

#     idx = 0
#     for size in sizes:
#         plot_distribution(data[idx:idx + size], ax=ax, **kwargs)
#         idx += size

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)

#     if path is not None:
#         fig.savefig(path, format='png')



def plot_train_test_split(train_data, test_data, path=None):
    '''Plot information about the train-test split.

    args:
        - train_data (pd.DataFrame)   
        - test_data (pd.DataFrame)   
    '''

    assert 'seq' in train_data.columns
    assert 'seq' in test_data.columns
    assert train_data.index.name == test_data.index.name == 'id'

    cmap = mpl.colormaps['Pastel2']

    # Things we care about are length distributions, as well as
    # proportion of negative and positive instances (and selenoproteins, for that matter)

    plt.figure(figsize=(10, 10))
    # Establish the subplots. First and second axes are pie charts, the third are length distributions. 
    axes = [plt.subplot(2, 2, 1), plt.subplot(2, 2, 2), plt.subplot(2, 2, (3, 4))]
    
    # Make the composition pie charts first. 
    for ax, data, title in zip(axes[:2], [train_data, test_data], ['Training set composition', 'Testing set composition']):
        sec_count, sec_trunc_count, other_count = 0, 0, 0

        for row in data.itertuples():
            # Yhid vonfiyion is met only for truncated selenoproteins. 
            if '[' in row.Index:
                sec_trunc_count += 1
            elif 'U' in row.seq:
                sec_count += 1
            else:
                other_count += 1

        ax.pie([sec_trunc_count, sec_count, other_count], labels=['trunc_sec', 'sec', 'normal'], autopct='%1.1f%%', frame=True, colors=cmap(np.arange(3)))
        ax.set_title(title)
        # Turn off annoying tick marks. 
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])

    data = {}
    data['train'] = np.array([len(s) for s in train_data['seq']])
    data['test'] = np.array([len(s) for s in test_data['seq']])
    # data = pd.DataFrame(data)
    # plot_distributions(lengths, ax=axes[2], xlabel='length', title='Sequence length distributions', path=None)
    # axes[2].legend(['train', 'test'])

    sns.histplot(data=data, ax=axes[2], legend=True, stat='proportion', multiple='dodge', bins=50, palette='Pastel2', ec=None)
    axes[2].set_xlabel('lengths')
    axes[2].set_ylabel('count')
    axes[2].set_title('Length distributions in training and test set')

    # Fix the layout and save the figure in the buffer.
    plt.tight_layout()
    plt.savefig(path, format='png')


def plot_filter_sprot(data, idxs=None, path=None):
    '''Visualizes the filtering procedure carried out by the filter_sprot
    function. Plots both the K-means clusters and the data which "passes"
    the filter.

    args:
        - data (np.array): An array containing the SwissProt embeddings before 
            being filtered.
        - idxs (np.array): The filter indices. 
        - path (str): The path for where to save the generated figure. 
    '''

    assert idxs is not None
    
    idxs = np.concatenate([np.arange(len(data)), idxs])
    n, m = len(data), len(idxs) # n is the number of non-filtered elements. 

    # Use PCA to put the data in two-dimensional space. 
    pca = sklearn.decomposition.PCA(n_components=2)
    data = pca.fit_transform(data.values)
    # Get the explained variance ration to determine how important each component is. 
    evrs = pca.explained_variance_ratio_

    # print('PCA decomposition of SwissProt data completed.')

    df = pd.DataFrame({f'PCA 1 (EVR={evrs[0]})':data[idxs, 0], f'PCA 2 (EVR={evrs[1]})':data[idxs, 1],  'filter':[0]*n + [1]*(m - n)})

    fig, ax = plt.subplots(1)
    # sns.scatterplot(data=df x='UMAP 1', y='UMAP 2', ax=ax, hue='label', legend=False, **kwargs)
    sns.scatterplot(data=df, x=f'PCA 1 (EVR={evrs[0]})', y=f'PCA 2 (EVR={evrs[1]})', ax=ax, hue='filter', legend=False, s=2, palette={0:'gray', 1:'black'})
    ax.set_title('Selecting representative sequences from SwissProt')

    fig.savefig(path, format='png')


# def plot_confusion_matrix(preds, labels, filename=None, title=None):
#     '''
#     Plots the confusion matrix for a set of predictions generated 
#     by a specific model. Because, in this case, the problem is one of
#     binary classification, this matrix displays true positives, false
#     positives, true negatives, and false negatives. 

#     args:
#         - preds (np.array)
#         - labels (np.array)
#     '''

#     # Confusion matrix whose i-th row and j-th column entry indicates 
#     # the number of samples with true label being i-th class and predicted 
#     # label being j-th class.
#     matrix = confusion_matrix(preds, labels)
#     # tn, fp, fn, tp = matrix.ravel()

#     fig, ax = plt.subplots(1)
#     sns.heatmap(matrix, ax=ax, annot=True)

#     ax.set_xlabel('true')
#     ax.set_ylabel('predicted')

#     # ax.set_xticklabels(['false positive', 'true positive'])
#     # ax.set_yticklabels(['true negative', 'false negative'])
#     ax.set_title(title)

#     if filename is not None:
#         fig.savefig(filename, format='png')




