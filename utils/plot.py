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

cmap = mpl.colormaps['Pastel2']
palette = 'Set2'

# TODO: Might be worth making an info object for plotting results of training. 


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


def plot_train_test_val_split(train_data, test_data, val_data, path=None):
    '''Plot information about the train-test split.

    args:
        - train_data (pd.DataFrame)   
        - test_data (pd.DataFrame)   
        - val_data (pd.DataFrame)
    '''

    # Things we care about are length distributions, as well as
    # proportion of negative and positive instances (and selenoproteins, for that matter)

    plt.figure(figsize=(15, 10))
    # Establish the subplots. First and second axes are pie charts, the third are length distributions. 
    axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2),plt.subplot(2, 3, 3), plt.subplot(2, 3, (4, 6))]
    titles = ['Training set composition', 'Testing set composition', 'Validation set composition']
    
    # Make the composition pie charts first. 
    for ax, data, title in zip(axes[:3], [train_data, test_data, val_data], titles):
        sec_count = np.sum([1 if '[' in row.Index else 0 for row in data.itertuples()])
        ax.pie([sec_count, len(data) - sec_count], labels=[f'truncated ({sec_count})', f'full-length ({len(data) - sec_count})'], autopct='%1.1f%%', colors=cmap(np.arange(3)))
        ax.set_title(title)

    data = {}
    data['train'] = np.array([len(s) for s in train_data['seq']])
    data['test'] = np.array([len(s) for s in test_data['seq']])
    data['val'] = np.array([len(s) for s in val_data['seq']])

    sns.histplot(data=data, ax=axes[-1], legend=True, stat='count', multiple='dodge', bins=50, palette='Pastel2', ec=None)
    axes[-1].set_yscale('log')
    axes[-1].set_xlabel('lengths')
    axes[-1].set_ylabel('log(count)')
    axes[-1].set_title('Length distributions')

    # Fix the layout and save the figure in the buffer.
    plt.tight_layout()
    plt.savefig(path, format='png')


def plot_train_(reporter, path=None, pool=True): # include=['train_loss', 'pooled_train_loss', 'val_loss']):
    '''Plots information provided in the info dictionary returned by the train_ model method.'''

    # fig, axes = plt.subplots(2, figsize=(12, 10), sharex=True)
    fig, axes = plt.subplots(2, figsize=(20, 10), sharex=True)

    # Seems as though model performance (as inficated by train loss) may vary substantially
    # between batches with and without a selenoprotein present. Might be helpful to label with this in mind. 

    # Pool the train accuracy and losses
    info = pool_train_info(info)

    loss_df = reporter.get_loss_info()
    acc_df = reporter.get_acc_info()

    sns.lineplot(data=loss_df, y='loss', x='batch', hue='metric', ax=axes[0], palette=palette)
    sns.lineplot(data=acc_df, y='accuracy', x='batch', hue='metric', ax=axes[1], palette=palette)

    axes[0].set_title(f"Weighted BCE loss (weight={info['bce_loss_weight']})")
    axes[0].set_yscale('log')
    # axes[0].set_xlabel('batch')

    axes[1].set_title('Accuracy')

    for ax in axes:
        ax.vlines(info['epoch_batches'], *ax.get_ylim(), linestyles='dotted', color='LightGray')
        ax.legend().set_title('')

    fig.savefig(path, format='png')


def plot_train_info(reporters, path=None, title='plot_train_losses', pool=True, sec_only=False):
    '''Takes a list of train outputs as input.'''

    fig, ax = plt.subplots(1, figsize=(16, 10))
    
    # Add bce_loss_weight information to each DataFrame. 
    dfs = [r.get_loss_info().assign(bce_loss_weight=r.bce_loss_weight) for r in reporters]
    df = pd.concat(dfs)
    
    metric = ('pooled_' if pool else '') + 'train_loss' + ('_batches_with_sec' if sec_only else '') 
    df = df[df['metric'] == metric]

    if not pool:
        sns.scatterplot(data=df, y='loss', x='batch', hue='bce_loss_weight', ax=ax, palette=palette, edgecolor=None, s=5)
    else:
        sns.lineplot(data=df, y='loss', x='batch', hue='bce_loss_weight', ax=ax, palette=palette)
    
    ax.legend(loc='upper right', title='bce_loss_weight') # Manually set legend location to avoid it being so slow. 

    ax.set_yscale('log')
    ax.set_title(title)

    fig.savefig(path, format='png')


# def plot_filter_sprot(data, idxs=None, path=None):
#     '''Visualizes the filtering procedure carried out by the filter_sprot
#     function. Plots both the K-means clusters and the data which "passes"
#     the filter.

#     args:
#         - data (np.array): An array containing the SwissProt embeddings before 
#             being filtered.
#         - idxs (np.array): The filter indices. 
#         - path (str): The path for where to save the generated figure. 
#     '''

#     assert idxs is not None
    
#     idxs = np.concatenate([np.arange(len(data)), idxs])
#     n, m = len(data), len(idxs) # n is the number of non-filtered elements. 

#     # Use PCA to put the data in two-dimensional space. 
#     pca = sklearn.decomposition.PCA(n_components=2)
#     data = pca.fit_transform(data.values)
#     # Get the explained variance ration to determine how important each component is. 
#     evrs = pca.explained_variance_ratio_

#     # print('PCA decomposition of SwissProt data completed.')

#     df = pd.DataFrame({f'PCA 1 (EVR={evrs[0]})':data[idxs, 0], f'PCA 2 (EVR={evrs[1]})':data[idxs, 1],  'filter':[0]*n + [1]*(m - n)})

#     fig, ax = plt.subplots(1)
#     # sns.scatterplot(data=df x='UMAP 1', y='UMAP 2', ax=ax, hue='label', legend=False, **kwargs)
#     sns.scatterplot(data=df, x=f'PCA 1 (EVR={evrs[0]})', y=f'PCA 2 (EVR={evrs[1]})', ax=ax, hue='filter', legend=False, s=2, palette={0:'gray', 1:'black'})
#     ax.set_title('Selecting representative sequences from SwissProt')

#     fig.savefig(path, format='png')


# Want to visualize selenoprotein enrichment in each cluster. Might be good to have some kind
# of heatmap, where the color indicates number of selenoproteins. 

# Actually just doing a scatter plot might be easier. Maybe just put scatter number on the x-axis, or
# do something with cluster size. OK, maybe cluster number on x-axis, size on the y-axis, and color corresponding
# to selenoprotein count. 

def plot_sample_kmeans_clusters(data, kmeans, n_clusters=None, sec_ids=None, path=None):
    '''Generates a plot showing the selenoprotein enrichment in the K-means clusters generated
    in the sample_kmeans_clusters function.'''

    fig, ax = plt.subplots(1, figsize=(14, 5))

    labels = np.arange(n_clusters) # The cluster labels!
    sizes = [np.sum(kmeans.labels_ == i) for i in labels] # Get the size of each cluster. 
    
    ids = data.index.to_numpy()
    sec_contents = np.array([np.sum(np.isin(ids[kmeans.labels_ == i], sec_ids)) for i in labels])

    df = {'cluster_label':labels, 'cluster_size':sizes, 'sec_content':sec_contents}
    ax = sns.scatterplot(ax=ax, data=df, hue='sec_content', x='cluster_label', y='cluster_size', palette=sns.color_palette("light:#5A9", as_cmap=True))
    sns.move_legend(ax, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    ax.set_title('Selenoprotein enrichment in K-means clusters')

    fig.savefig(path, format='png')

def plot_get_homology_cluster():
    pass



