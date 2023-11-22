from Bio import Phylo
# import dendropy I actually don't think I need dendropy. 
from utils import *
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, NoReturn
from Bio.Phylo.BaseTree import Tree, Clade
# import pydot

BACTERIA_TREE_PATH = load_config_paths()['bacteria_tree_path']
ARCHAEA_TREE_PATH = load_config_paths()['archaea_tree_path']

# Some specs to make sure everything is in line with Nature Micro requirements.
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 10
FIGSIZE = (4, 3)
FONT = 'Arial'
PALETTE = 'Set1'

# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':[FONT], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})
# plt.rc('text',  **{'usetex':True}) # Enable use of LaTeX

LEVEL_ABBREVIATIONS = {'phylum':'p', 'domain':'d', 'genus':'g', 'species':'s', 'order':'o', 'c':'class', 'family':'f'}


def tree_label_terminals_with_taxonomy(tree, level='c'):
    '''Label terminals such that all terminals that belong to the same genus, order, family, etc. have the same name,
    corresponding to the taxonomical categorie to which they all belong.'''
    
    # pattern = f'\d+\.\d:[\w__; ]*{level}__\w+; [\w__; ]*[\w__]+' # Pattern to match. 
    pattern = f'.*{level}__.+' # Pattern to match. 

    for node in tree.find_elements(name=pattern, terminal=False):
        t = re.search(f'{level}__\w+', node.name).group(0)
        # .group(0) # Extract the texonomical category from the matching string.
        for leaf in node.get_terminals():
            leaf.name = t[3:] # Set the leaf name to the taxonomy. 
    return tree

def tree_collapse_nodes(tree):
    '''Iterate over tree nodes, ensuring that each terminal node has a unique name. This should be called after
    tree_label_terminals_with_taxonomy, so that it effectively collapses everything in the same taxonomical group.'''

    for node in tree.find_clades():
        if node.count_terminals() > 1: # Check if the node is followed by more than one leaf. 
            leafs = node.get_terminals()
            if len(set([l.name for l in leafs])) == 1: # All the leafs in this clade have the same name.
                for leaf in leafs[1:]: # Remove all nodes but the first.    
                    # print(leaf.name)                            
                    tree.prune(leaf)
    for node in tree.get_terminals():
        if 's__' in node.name:
            tree.prune(node)

    return tree


def tree_remove_non_terminal_labels(tree:Tree) -> Tree:
    '''Remove the labels from any non-terminal tree node, so plotting is nicer. This should only be called
    after tree_collapse_nodes.'''

    for node in tree.find_clades(terminal=False):
        node.name = None
    return tree


def tree_draw(tree:Tree, 
    path:str=None,
    relative_selenoprotein_content:Dict[Tree, float]={}) -> NoReturn:

    def get_x_positions(tree):
        '''Create a mapping of each clade to its horizontal position.'''
        # Assume unit branch lengths for simplicity. 
        return tree.depths()

    def get_y_positions(tree):
        '''Create a mapping of each node to its vertical position.'''
        max_height = tree.count_terminals()
        # Why are the terminals reversed here?
        ys = {leaf: max_height - i for i, leaf in enumerate(reversed(tree.get_terminals()))}

        # Internal nodes are place at midpoint of children.
        def get_y(node):
            '''Recursively calculate the y position for each internal node. Positions of the leaf nodes have already been calculated.'''
            for child in node: # Didn't know you could iterate over clades like this!
                if child not in ys: # When it hits the root node, this will evaluate to False. 
                    get_y(child)
            first_child, last_child = node.clades[0], node.clades[-1]
            ys[node] = (ys[first_child] + ys[last_child]) / 2.0

        get_y(tree.root)
        return ys

    xs, ys = get_x_positions(tree), get_y_positions(tree)
    fig, ax = plt.subplots(1)

    lw, c = '0.1', 'black' # Just stick to this for now. 

    # NOTE: Calling iter on a node iterates through the direct descendents.
    def draw_node(node):
        '''Recursively draw a tree from the root node.'''
        if not node.is_terminal(): # Don't try to call the function again if terminal.
            # Get the position of the first verical line, halway between parent and closest child node. 
            y_min, y_max = min([ys[n] for n in node.clades]), max([ys[n] for n in node.clades])
            x = (min([xs[n] for n in node.clades]) + xs[node]) / 2.0
            ax.vlines(x, y_min, y_max, 'gray')
            # Draw a horizontal line between the parent and first vertical line. 
            ax.hlines(ys[node], xs[node], x, 'gray')
            # Draw a horizontal line between the first vertical line and each child node.
            for child in node.clades:
                ax.hlines(ys[child], x, xs[child], 'gray')
                draw_node(child) # Recursive call. 
    
    draw_node(tree.root)

    data, colors = [], []
    for leaf in tree.get_terminals():
        x, y = xs[leaf], ys[leaf]
        data.append([x, y])
        colors.append(relative_selenoprotein_content.get(leaf, 'lightblue'))
        ax.text(x, y, leaf.name, fontsize=10)

    ax.scatter(*np.array(data).T, s=30, c=colors)

    ax.axis('off')

    if path:
        fig.savefig(path, dpi=500, format='png')


if __name__ == '__main__':

    level = 'phylum'


    tree = Phylo.read(ARCHAEA_TREE_PATH, 'newick')

    tree = tree_label_terminals_with_taxonomy(tree, level=LEVEL_ABBREVIATIONS[level])
    tree = tree_collapse_nodes(tree)
    # tree = tree_remove_non_terminal_labels(tree)

    print(tree.count_terminals(), f'leaf nodes present in the final {level} tree.')

    tree_draw(tree, path='tree.png')









