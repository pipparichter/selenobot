'''
Plotting utilities for the ProTex tool. 
'''

import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns




def plot_umap(data, filename='umap_plot.png'):
    '''
    Apply UMAP dimensionality reduction to embeddings, and plot in
    two-dimensional space. 
    '''
    reducer = umap.UMAP()
    fig, ax = plt.subplots(1)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    legend = []

    for label, embeddings in data.items():

        legend.append(label)

        # Need to convert from tensor back to NumPy array, I think.
        reduced_embeddings = reducer.fit_transform(embeddings)
        # print(reduced_embeddings.shape, reduced_embeddings_truncated.shape)
        reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['UMAP 1', 'UMAP 2'])

        sns.scatterplot(reduced_embeddings, ax=ax, x='UMAP 1', y='UMAP 2')
    
    ax.legend(legend)
    plt.savefig(filename)

# TODO: It might be useful to visualize clusters somehow. Maybe show how they are represented in UMAP
# space... Only issue is that there are over 8000 clusters, so this may not be useful or worthwhile. 