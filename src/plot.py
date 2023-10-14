import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import matplotlib as mpl
import sklearn
import reporter
import re

import utils
from tqdm import tqdm

from typing import NoReturn, Tuple, List, Dict

import ete3
import PyQt5
from ete3 import NodeStyle
# from Bio import Phylo
import io

# Some specs to make sure everything is in line with Nature Micro requirements. 
DPI = 500
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7



# TODO: Might be worth making an info object for plotting results of training. 

# TODO: Need to decide on a color scheme. Will probably go with blues. 

def set_fontsize(ax:plt.Axes, legend=True) -> NoReturn:
    ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)
    ax.set_title(ax.get_title(), fontsize=TITLE_FONT_SIZE)
    
    if legend:
        ax.legend(fontsize=LABEL_FONT_SIZE)


def get_palette(n_colors:int):
    '''Get the color palette from the matplotlib Blues colorset.'''
    return sns.color_palette('Blues', n_colors=2)


# Josh said that simply having a three-bar plot, each representing a different section of the dataset, 
# and each bar stacked to indicate sec content, should be sufficient. 
def plot_train_test_val_split(
    train_data:pd.DataFrame=None, 
    test_data:pd.DataFrame=None, 
    val_data:pd.DataFrame=None, 
    title:str='plot.plot_train_test_val_split',
    path:str=None) -> None: 
    '''Plot information about the train-test-validation  split.

    args:
        - train_data: A DataFrame containing the train data.   
        - test_data: A DataFrame containing the test data.     
        - val_data: A DataFrame containing the validation data.  
        - title: A title for the plot. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
    '''

    # Things we care about are length distributions, as well as
    # proportion of negative and positive instances (and selenoproteins, for that matter)

    fig, ax = plt.subplots(1, figsize=(6, 5)) #, figsize=(15, 10))

    plot_data = {'dataset':['train', 'test', 'val']}
    plot_data['truncated'] = [np.sum([1 if '[' in row.Index else 0 for row in data.itertuples()]) for data in [train_data, test_data, val_data]] 
    plot_data['full_length'] = [len(data) - count for data, count in zip([train_data, test_data, val_data], plot_data['truncated'])]

    plot_data = pd.DataFrame(plot_data).set_index('dataset')
    plot_data.plot(kind='bar', ax=ax, color=['cornflowerblue', 'lightsteelblue'], edgecolor='black', linewidth=0.5)

    sizes = [len(train_data), len(test_data), len(val_data)]
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{np.round(100 * x/y, 1)}%' for x, y in zip(container.datavalues, sizes)], fontsize=LABEL_FONT_SIZE)

    # Make sure all the labels look how I want, and are the correct size. 
    ax.set_title(title)
    ax.set_ylabel('count')
    ax.set_xlabel('') # Remove the x label.

    ax.legend(fontsize=LABEL_FONT_SIZE)

    # # Establish the subplots. First and second axes are pie charts, the third are length distributions. 
    # axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2),plt.subplot(2, 3, 3), plt.subplot(2, 3, (4, 6))]
    # titles = ['Training set composition', 'Testing set composition', 'Validation set composition']
    
    # # Make the composition pie charts first. 
    # for ax, data, title in zip(axes[:3], [train_data, test_data, val_data], titles):
    #     sec_count = np.sum([1 if '[' in row.Index else 0 for row in data.itertuples()])
    #     ax.pie([sec_count, len(data) - sec_count], labels=[f'truncated ({sec_count})', f'full-length ({len(data) - sec_count})'], autopct='%1.1f%%', colors=cmap.resampled(3)(np.arange(3)))
    #     ax.set_title(title)

    # data = {}
    # data['train'] = np.array([len(s) for s in train_data['seq']])
    # data['test'] = np.array([len(s) for s in test_data['seq']])
    # data['val'] = np.array([len(s) for s in val_data['seq']])

    # sns.histplot(data=data, ax=axes[-1], legend=True, stat='count', multiple='dodge', bins=50, palette=palette, ec=None)
    # axes[-1].set_yscale('log')
    # axes[-1].set_xlabel('lengths')
    # axes[-1].set_ylabel('log(count)')
    # axes[-1].set_title('Length distributions')

    # # Fix the layout and save the figure in the buffer.
    # plt.tight_layout()
    
    if path is not None:
        plt.savefig(path, format='png', dpi=DPI)


def plot_train_curve(reporter:reporter.Reporter, path:str=None, title:str='plot.plot_train_curve', pool=True) -> NoReturn: 
    '''Plots information provided in the Reporter object returned by the train_ model method.
    
    args:
        - reporter: The reporter object containing the train information. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
    '''

    fig, ax = plt.subplots(1)
    
    # # Add horizontal lines indicating epochs. 
    # ax.vlines(reporter.get_epoch_batches(), *ax.get_ylim(), linestyles='dotted', color='LightGray')
    x = 'batch' if not pool else 'epoch'
    sns.lineplot(data=reporter.get_loss_info(pool=pool), y='value', x=x, hue='metric', ax=ax, palette=get_palette(2))
    
    ax.legend().set_title('') # Turn off legend title because it looks crowded. 

    # Make sure all labels are the right size. 
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_ylabel('log(loss)')

    set_fontsize(ax)

    # fig, axes = plt.subplots(2, figsize=(16, 10), sharex=True)
    
    # # NOTE: Don't need to plot accuracy on the training curve. 
    # loss_df = reporter.get_loss_info()
    # acc_df = reporter.get_acc_info()

    # sns.lineplot(data=loss_df, y='value', x='batch', hue='metric', ax=axes[0])
    # sns.lineplot(data=acc_df, y='value', x='batch', hue='metric', ax=axes[1])

    # axes[0].set_title(f"Weighted BCE loss (weight={reporter.bce_loss_weight})")
    # axes[0].set_yscale('log')
    # axes[0].set_ylabel('log(loss)')

    # axes[1].set_title('Accuracy')
    # axes[1].set_ylabel('accuracy')

    # for ax in axes:
    #     ax.vlines(reporter.get_epoch_batches(), *ax.get_ylim(), linestyles='dotted', color='LightGray')
    #     ax.legend().set_title('')

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)



def plot_confusion_matrix(reporter:reporter.Reporter, path:str=None, title:str='plot.plot_confusion_matrix', ax:plt.Axes=None) -> NoReturn:
    '''Plots a confusion matrix using a Reporter.
    
    args:
        - reporter: The reporter object containing the train information. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - ax: A matplotlib axis. If specified, the confusion matrix is added to this axis. 
    '''

    # Extract the confusion matrix from the reporter object. 
    (tn, fp, fn, tp) = reporter.get_confusion_matrix()
    # Convert the heatmap to a pandas DataFrame for the sake of labels. 

    # Confusion matrix function takes y_predicted and y_true as inputs, which is exactly the output of the predict method.
    if ax is None:
        fig, ax = plt.subplots(1)
        cbar = True
        labels = [[f'true negative ({tn})', f'false positive ({fp})'], [f'false negative ({fn})', f'true positive ({tp})']]
    else: # If the matrix is being plot on an inset axis, shorten the labels and remove the colorbar so it doesn't look crowded.
        cbar = False
        labels = [[f'TN ({tn})', f'FP ({fp})'], [f'FN ({fn})', f'TP ({tp})']]

    # (tn, fp, fn, tp)
    annot_kws = {'fontsize':LABEL_FONT_SIZE}
    sns.heatmap([[tn, fp], [fn, tp]], fmt='', annot=labels, annot_kws=annot_kws, ax=ax, cmap=mpl.colormaps['Blues'], cbar=False, linewidths=0.5, linecolor='black')

    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    # Make the lines around the confusion matrix visible. 
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    # Make sure all fonts are correct. 
    set_fontsize(ax, legend=False)
    
    if path is not None:
        fig.save(path, format='png', dpi=DPI)


def rgb_to_hex(r, g, b):
    '''Convert a red, blue, and green value (the output of map.to_rgba) to a hex string.'''
    # First need to put each value into hexadecimal. 
    return '#' + '%02x%02x%02x' % (r, g, b)


def get_taxonomy_to_sec_content_map(gtdb_data, genome_id_to_taxonomy_map:Dict[str, str]=None):
    '''Use GTDB taxomonomy to combine different families and plot average selenoprotein content
    across labeled members.

    args:
        - gtdb_data 

    '''
    assert genome_id_to_taxonomy_map is not None, 'plot.get_class_to_sec_content: A dictionary mapping genome ID to class name must be specified.'
    
    # Get a list of all possible taxonomical categories at the specified level. 
    keys = np.unique([t for t in  genome_id_to_taxonomy_map.values()])
    taxonomy_to_sec_content_map = {t:[] for t in keys}

    for row in gtdb_data.itertuples():
        # Get the relevant taxonomy. 
        try: # Try to extract the specified taxonomic label. 
            taxonomy = genome_id_to_taxonomy_map.get(row.Index, None)     
            sec_content = row.total_hits / row.total_genes
            taxonomy_to_sec_content_map[taxonomy].append(sec_content)
        except KeyError: # If the taxonomy is not found, move on to the next iteration. 
            continue

    # Take the average of each sec content list. 
    # Possibly will need to make sure that we are not trying to take the mean of an empty list.
    return {t:0 if np.isnan(np.mean(s)) else np.mean(s) for t, s in taxonomy_to_sec_content_map.items()}


def get_genome_id_to_taxonomy_map(taxonomy_data:pd.DataFrame, level:str='class') -> Dict[str, str]:
    '''Processes a taxonomy file, converting it into a dictionary mapping genome ID
    to the name of the family to which the genome belongs.

    args:
        - taxonomy_data: Contains two columns, genome_id and taxonomy. The taxonomy column contains
            strings with each taxonomical level separated by semicolons. 
        - level: The taxonomic level for which to accumulate data. 
    '''
    # Collect information into a dictionary mapping 
    genome_id_to_taxonomy_map = {}
    for row in taxonomy_data.itertuples():
        try:
            genome_id_to_taxonomy_map[row.genome_id] = parse_taxonomy(row.taxonomy)[level]
        except KeyError:
            print(f'plot.get_genome_id_to_taxonomy_map: Genome {genome_id} does not have specified taxonomy data.')
            continue

    return genome_id_to_taxonomy_map


def parse_taxonomy(taxonomy:str) -> Dict[str, str]:
    '''Extract information from a taxonomy string.'''

    m = {'o':'order', 'd':'domain', 'p':'phylum', 'c':'class', 'f':'family', 'g':'genus', 's':'species'}

    parsed_taxonomy = {}
    # Split taxonomy string along the semicolon...
    for x in taxonomy.strip().split(';'):
        f, data = x.split('__')
        parsed_taxonomy[m[f]] = data
    
    return parsed_taxonomy # Return None if flag is not found in the taxonomy string. 

    

# def label_nodes_with_taxonomy(tree, genome_id_to_family_map=None):
def label_nodes_with_taxonomy(tree:ete3.Tree, genome_id_to_taxonomy_map:Dict[str, str]=None) -> NoReturn:
    '''Label all nodes in a tree with their family. Labels according to the specified taxonomic level.
    
    args:
        - tree: A Tree object with unlabeled nodes. 
        - genome_id_to_taxonomy_map: A map co
    '''
    
    assert genome_id_to_taxonomy_map is not None, 'plot.get_class_to_sec_content: A dictionary mapping genome ID to class name must be specified.'

    for leaf in tree.get_leaves():
        # family = genome_id_to_family_map[node.name]
        taxonomy = genome_id_to_taxonomy_map[leaf.name]
        leaf.add_features(taxonomy=taxonomy)




def is_leaf(node):
    '''Function for collapsing the tree according to taxonomy.'''
    # If a node is monophyletic for the taxonomy, then it is a leaf.
    leaves = node.get_leaves()
    taxonomy = leaves[0].taxonomy
    if np.all([taxonomy == l.taxonomy for l in leaves]):
        node.name = taxonomy
        return True
    else:
        return False


# def merge_nodes_by_family(tree:ete3.Tree, families=List[str], genome_id_to_family_map:Dict[str,str]=None) -> ete3.Tree :
def merge_nodes_by_taxonomy(tree:ete3.Tree, classes=List[str], genome_id_to_taxonomy_map:Dict[str,str]=None) -> ete3.Tree :
    '''Takes a tree as input, and merges all the nodes so that there is only one node for
    a particular family.'''

    # Annotate all nodes with stored taxonomical data. 
    label_nodes_with_taxonomy(tree, genome_id_to_taxonomy_map=genome_id_to_taxonomy_map)
    return ete3.Tree(tree.write(is_leaf_fn=is_leaf), format=1, quoted_node_names=True)


def plot_gtdb_tree(
        tree_path:str, 
        gtdb_data:pd.DataFrame, 
        taxonomy_path:str=None, 
        level:str='class',
        sec_content_threshold:float=None,
        path:str=None, 
        show_sec_content:bool=True,
        font_size:int=15, 
        margin:int=10) -> NoReturn:
    '''Phylogenetic tree from the GTDB database with selenoprotein content coded as a color. 
    Selenoprotein content is given as total selenoproteins divided by total genes.'''

    # assert path is not None, 'plot.plot_gtdb_tree: An output path must be specified.'
    assert taxonomy_path is not None, 'plot.plot_gtdb_tree: Path to taxonomy file must be specified.'
    
    # Ready in te taxonomy data. 
    taxonomy_data = pd.read_csv(taxonomy_path, names=['genome_id', 'taxonomy'], delimiter='\t')

    # Read in the tree data from the file. Should be in paranthetical format. 
    with open(tree_path, 'r') as f:
        tree_data = f.read()


    genome_id_to_taxonomy_map = get_genome_id_to_taxonomy_map(taxonomy_data, level=level)
    taxonomy_to_sec_content_map = get_taxonomy_to_sec_content_map(gtdb_data, genome_id_to_taxonomy_map=genome_id_to_taxonomy_map)

    
    print(f'plot.plot_gtdb_tree: {len(np.unique(list(genome_id_to_taxonomy_map.values())))} {level} represented in taxonomy_to_sec_content_map.')
    
    tree = ete3.Tree(tree_data, format=1, quoted_node_names=True) 
    tree = merge_nodes_by_taxonomy(tree, genome_id_to_taxonomy_map=genome_id_to_taxonomy_map)

    # Create a normalized colormap for coloring the nodes. 
    cmap = matplotlib.colormaps['Blues']
    all_sec_contents = gtdb_data['total_hits'].values / gtdb_data['total_genes'].values
    norm = matplotlib.colors.Normalize(vmin=min(list(taxonomy_to_sec_content_map.values())), vmax=max(list(taxonomy_to_sec_content_map.values())))
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Color the merged tree according to sec_content.
    for node in tree.traverse():
        
        # Make sure the node bubbles aren't visible -- maybe a cleaner way to do this. 
        node_style = ete3.NodeStyle()
        node_style['size'] = 0
        node_style['hz_line_width'] = 2
        node_style['hz_line_width'] = 2
        node.set_style(node_style)

        if node.is_leaf():

            sec_content = np.round(taxonomy_to_sec_content_map[node.name], 2)

            # If a sec_content threshold is specified, remove all leaves which do not meet the threshold.
            if sec_content_threshold is not None:
                if sec_content < sec_content_threshold:
                    node.delete()
                    continue

            label = node.name + f' ({sec_content})' if show_sec_content else node.name

            r, g, b, _ = m.to_rgba(sec_content, bytes=True) # Set bytes=True to get values between 0 and 255. 
            if (r > 200 )or (g > 200) or (b > 200): # I think a reasonable test for if the background is dark?
                font_color = '#000000'
            else:
                font_color = '#FFFFFF'

            
            face = ete3.TextFace(text=label, ftype='arial', fgcolor=font_color, fsize=font_size)
            face.background.color = rgb_to_hex(r, g, b)
            face.border.type = 0 # Make the border solid. 
            face.border.width = 1
            face.penwidth = 2
            face.margin_left = margin
            face.margin_right = margin
            face.margin_top = margin
            face.margin_bottom = margin

            # ete3.add_face_to_node(face, node, aligned=True, column=0) # Not sure what the column parameter does. 
            node.add_face(face, column=0) # , position='aligned')

    tree_style = ete3.TreeStyle()

    tree_style.mode = 'c'
    tree_style.arc_start = -180 # 0 degrees = 3 o'clock
    tree_style.arc_span = 180
    tree_style.show_leaf_name = False
    tree_style.show_scale = False

    print(f'plot.plot_gtdb_tree: {len(tree.get_leaves())} leaves in final tree.')

    tree.render(path, tree_style=tree_style)


def plot_roc_curve(
        reporters:List[reporter.Reporter], 
        path:str=None, 
        title:str='plot.plot_roc_curve', 
        ax:plt.Axes=None,
        add_confusion_matrix:bool=False) -> NoReturn:
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    args:
        - reporters: A list of reporters containing test information.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - ax: A matplotlib axis. If specified, the confusion matrix is added to this axis. 
        - add_confusion_matrix: Whether or not to include the confusion matrix as an inset. 
    '''

    if ax is None:
        fig, ax = plt.subplots(1)

    # First need to organize the information into a DataFrame for plotting. 
    data = {'true_positive_rate':[r.get_true_positive_rate() for r in reporters], 'false_positive_rate':[r.get_false_positive_rate() for r in reporters]}
    # Not totally sure why I am getting NaNs here...
    data = pd.DataFrame(data) # .fillna(0)

    # NOTE: Threshold is an upper bound, so when threshold is 1, everything should be classified as 0. When threshold
    # is zero, everything should be classified as 1.

    sns.lineplot(data=data, y='true_positive_rate', x='false_positive_rate', ax=ax, color='cornflowerblue', legend=None)

    if add_confusion_matrix:
        # Inset axes in the plot showing the confusion matrix. Should be in the bottom right corner. 
        # NOTE: (0,0) is bottom left and (1,1) is top right of the axes. This is the pixel coordinate system of the display. (0,0) is the bottom left and (width, height) is the top right of display in pixels.
        axins = ax.inset_axes([0.5, 0.1, 0.4, 0.5]) #, edgecolor='black')
        plot_confusion_matrix(reporters[3], title='', ax=axins)

    ax.set_title(title)

    # Make sure font sizes are correct. 
    set_fontsize(ax, legend=False)

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)



def plot_roc_curve_comparison(
        reporters:Tuple[List[reporter.Reporter], List[reporter.Reporter]],
        labels:Tuple[str, str]=('EmbeddingClassifier', 'AacClassifier'),
        path:str=None, 
        title:str='plot.plot_roc_curve_comparison') -> NoReturn: 
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    thargs:
        - reporters: A tuple of lists of reporters containing test information.
        - labels: The names of the two ROC curves being compared.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
    '''
    fig, ax = plt.subplots(1)

    # The first set of reporters specified should be the baseline. 
    plot_roc_curve(reporters[0], add_confusion_matrix=False, title='', ax=ax, path=None)
    # Add the inset confusion matrix for the non-baseline ROC curve. 
    plot_roc_curve(reporters[1], add_confusion_matrix=True, title='', ax=ax, path=None)

    # Should be two lines on the plot, and the first should be the baseline. 
    ax.lines[0].set_linestyle('--')
    ax.lines[0].set_color('gray')

    ax.set_title(title)

    # Make sure font sizes are correct. 
    set_fontsize(ax)

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)

