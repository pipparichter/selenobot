from Bio import Phylo
# import dendropy I actually don't think I need dendropy. 
from utils import *
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, NoReturn, Tuple
from Bio.Phylo.BaseTree import Clade
# import pydot
import copy
import pickle

# What are the RS, GB, etc. prefixes at the front of the genome IDs?

BACTERIA_TREE_PATH = load_config_paths()['bacteria_tree_path']
ARCHAEA_TREE_PATH = load_config_paths()['archaea_tree_path']
BACTERIA_METADATA_PATH = load_config_paths()['bacteria_metadata_path']
ARCHAEA_METADATA_PATH = load_config_paths()['archaea_metadata_path']
# BACTERIA_TAXONOMY_PATH = load_config_paths()['bacteria_taxonomy_path']
# ARCHAEA_TAXONOMY_PATH = load_config_paths()['archaea_taxonomy_path']


# Some specs to make sure everything is in line with Nature Micro requirements.
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 10
FONT = 'Arial'
PALETTE = 'Set1'

# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':[FONT], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})
# plt.rc('text',  **{'usetex':True}) # Enable use of LaTeX

def read(path:str, format='newick'):
    '''Read in a new Tree object from a file.'''
    tree = Phylo.read(path, format)
    return Tree(tree)


class TreeRank():
    '''A class for cleaning up how tree granularity is handled.'''
    ordering = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    names = {'d':'domain', 'p':'phylum', 'c':'class', 'o':'order', 'f':'family', 'g':'genus', 's':'species'}
    
    def __init__(self, r):
        '''Initialized a TreeResolution object.'''
        f ='tree.TreeRank.__init__'

        assert r in TreeRank.names.keys(), f'{f}: Specified taxonomical rank abbreviation {r} is invalid.'

        self.r = r
        self.rank_name = TreeRank.names[r]
        self.rank_order = TreeRank.ordering.index(self.rank_name)

    def __lt__(self, other):
        '''Overloading less than operator.'''
        return self.rank_order < other.rank_order

    def __gt__(self, other):
        '''Overloading greater than operator.'''
        return self.rank_order > other.rank_order

    def __eq__(self, other):
        '''Overloading equals operator.'''
        return other.rank_order == self.rank_order

    def __repr__(self):
        '''String representation.'''
        return f'TreeRank({self.rank_name})'
    
    def __str__(self):
        '''String representation.'''
        return self.r


class Tree(Phylo.BaseTree.Tree):
    '''Adding functionality to the BioPython Tree class.'''
    # Pattern for matching strings with numbers, letters (upper and lower case), whitespace, and dashes.
    # I think this covers all possible names. 
    node_name_pattern = '[a-zA-Z0-9-\s]+'

    def __init__(self, tree:Phylo.BaseTree, rank_leaves:str='s', rank_root:str='d'):

        # super(Tree, self).__init__(root=copy.deepcopy(tree.clade()), rooted=True)
        # Set the name of the tree equal to the taxonomy at the root. 
        name = re.search(f'{rank_root}__([\w]+)', tree.root.name).group(1)
        super(Tree, self).__init__(root=tree.root, rooted=True, name=name) # tree.clade() just returns the root. 
        
        assert rank_root < rank_leaves, f'tree.Tree.__init__: The rank of the root, {repr(rank_root)}, must be higher than {repr(rank_leaves)}.'
        # Initialize the leaf rank to species level, and the root rank to domain. 
        self.rank_root = TreeRank(rank_root)
        self.rank_leaves = TreeRank(rank_leaves)

        self.original = copy.deepcopy(tree.root) # Store a copy of the original tree. 

    def new_root(self, **kwargs):
        '''Scan the tree for the node which matches the taxonomy specified by the keyword arguments. Returns a new
        Tree object with a new Node 
        
        args:
            - tree: The tree to re-root. 
            - kwargs: Specifying the taxonomical level and category of the new root, e.g. p='Methanobacteriota'.
        '''
        f = 'tree.tree_find_new_root'
        assert len(kwargs) > 0, f'{f}: Specifications for a new node must be given. '
        assert len(kwargs) < 2, f'{f}: Only one new node can be set.'
        
        self.reset()

        rank, taxonomy = list(kwargs.items())[0]

        # pattern = f'\d+\.\d:[\w__; ]*{level}__\w+; [\w__; ]*[\w__]+' # Pattern to match. 
        pattern = f'.*{rank}__{taxonomy}.*' # Pattern to match. 
        print(pattern)

        # The tree is searched depth-first. This means that the first element returned is the one closest to the root. 
        # find_any returns the first element found which matches the criterion. 
        new_root = self.find_any(name=pattern)
        assert new_root is not None, f'{f}: Node {taxonomy} specified to be the new root is was not found in the tree.'

        # Creates a deep copy of the original tree. 
        self.set_rank_root(TreeRank(rank))
        self.name = re.search(f'{self.rank_root.r}__({Tree.node_name_pattern})', new_root.name).group(1)
        self.root = new_root

    def set_rank_leaves(self, rank:TreeRank):
        '''Set a new taxonomical rank for the tree's leaves.'''
        assert self.rank_root < rank, f'tree.Tree.set_rank_root: The rank of the root, {repr(self.rank_root)}, must be higher than {repr(rank)}.'
        self.rank_leaves = rank
    
    def set_rank_root(self, rank:TreeRank):
        '''Set a new taxonomical rank for the tree's leaves.'''
        assert self.rank_root < rank, f'tree.Tree.set_rank_root: The rank of the root, {repr(self.rank_root)}, must be higher than {repr(rank)}.'
        self.rank_root = rank

    def reset(self):
        '''Reset the tree to the original tree loaded from the file.'''
        # tree = copy.deepcopy(self) # Copy the tree. 
        self.root = copy.deepcopy(self.original) # Copy the original tree over to the current root. 
        self.rank_root = TreeRank('d')
        self.rank_leaves = TreeRank('s')
        
        name = re.search(f'{self.rank_root}__({Tree.node_name_pattern})', tree.root.name).group(1)
        self.name = name # Remove the {r}__ prefix from the name. 

    def collapse(self, rank:str):
        '''Collapse tree nodes to a lower taxonomical resolution. This function creates a copy of the tree with collapsed leaves.'''
        self.set_rank_leaves(TreeRank(rank)) # Set a new rank. 
        self.label_terminals_by_rank() # Re-label all terminals to match new rank resolution. 

        # Iterate over tree nodes, making sure the name of each terminal is unique.
        for node in self.find_clades(): # find_clades does not include terminals.
            if node.count_terminals() > 0: # Check if the node is followed by more than one leaf. 
                leaves = node.get_terminals()
                if len(set([l.name for l in leaves])) == 1: # All the leafs in this clade have the same name.
                    for leaf in leaves[1:]: # Remove all nodes but the first.                              
                        self.prune(leaf)


    def label_terminals_by_rank(self):
        '''Label terminals such that all terminals that belong to the same genus, order, family, etc. have the same name,
        corresponding to the taxonomical category to which they all belong.'''
        
        # pattern = f'\d+\.\d:[\w__; ]*{level}__\w+; [\w__; ]*[\w__]+' # Pattern to match. 
        pattern = f'.*{self.rank_leaves}__.*' # Pattern to match. 
        print(pattern)

        # for node in self.find_elements(name=pattern): #), terminal=False):
        for node in self.find_elements(name=pattern): #), terminal=False):
            name = re.search(f'{tree.rank_leaves}__({Tree.node_name_pattern})', node.name).group(1)
            # .group(0) # Extract e texonomical category from the matching string.
            if node.is_terminal():
                node.name = name
            for leaf in node.get_terminals():
                leaf.name = name # Set the leaf name to the taxonomy. 
        
        # There seems to be some stuff missing in the tree file. 
        for leaf in self.get_terminals():
            # Check to see if a {r}__ prefix is still present in the leaf's name.
            name = re.search(f'\w__({Tree.node_name_pattern})', leaf.name)
            if name is not None:
                # leaf.name = name.group(1)
                leaf.name = ''
                print(f'tree.Tree.label_terminals_by_rank: No matching taxonomy of level {repr(self.rank_leaves)} was found for {name.group(1)}')

    def draw(self, 
        path:str=None,
        colors:Dict[str, float]={}) -> NoReturn:
        '''Plot the current tree, i.e. the one stored in root.'''
        
        # Going to want to scale the figure size according to the number of terminals. 
        # Matplotlib figure size is in inches by default. Font size is 10 by default, which we will round to 0.15 in. 
        height = self.count_terminals() * 0.15 * 2 # Multiply by 2 to give extra space. 
        # width = 0.5 * self.total_branch_length() # Sum up all the branch lengths and scale so the figure is not huge. 
        fig, ax = plt.subplots(1, figsize=(6, height))

        def get_x_positions():
            '''Create a mapping of each clade to its horizontal position.'''
            # Assume unit branch lengths for simplicity. 
            return self.depths()

        def get_y_positions():
            '''Create a mapping of each node to its vertical position.'''
            max_height = self.count_terminals()
            # Why are the terminals reversed here?
            ys = {leaf: max_height - i for i, leaf in enumerate(reversed(self.get_terminals()))}

            # Internal nodes are place at midpoint of children.
            def get_y(node):
                '''Recursively calculate the y position for each internal node. Positions of the leaf nodes have already been calculated.'''
                for child in node: # Didn't know you could iterate over clades like this!
                    if child not in ys: # When it hits the root node, this will evaluate to False. 
                        get_y(child)
                
                first_child, last_child = node.clades[0], node.clades[-1]
                ys[node] = (ys[first_child] + ys[last_child]) / 2.0

            get_y(self.root)
            return ys

        xs, ys = get_x_positions(), get_y_positions()

        lw, c = 1, 'black' # Just stick to this for now. 

        # NOTE: Calling iter on a node iterates through the direct descendents.
        def draw_node(node):
            '''Recursively draw a tree from the root node.'''
            if not node.is_terminal(): # Don't try to call the function again if terminal.
                # Get the position of the first verical line, halway between parent and closest child node. 
                y_min, y_max = min([ys[n] for n in node.clades]), max([ys[n] for n in node.clades])
                x = (min([xs[n] for n in node.clades]) + xs[node]) / 2.0
                ax.vlines(x, y_min, y_max, 'gray', lw=lw)
                # Draw a horizontal line between the parent and first vertical line. 
                ax.hlines(ys[node], xs[node], x, 'gray', lw=lw)
                # Draw a horizontal line between the first vertical line and each child node.
                for child in node.clades:
                    ax.hlines(ys[child], x, xs[child], 'gray', lw=lw)
                    draw_node(child) # Recursive call. 
        
        draw_node(self.root)

        data = []
        # Does get_terminals() return the same list each time? Probably yes. 
        for leaf in self.get_terminals():
            x, y = xs[leaf], ys[leaf]
            data.append([x, y])
            ax.text(x + 0.01, y - 0.2, leaf.name, fontsize=10)

        colors = [colors.get(leaf.name, 'lightblue') for leaf in self.get_terminals()]
        ax.scatter(*np.array(data).T, s=50, c=colors)

        ax.axis('off')
        ax.set_title(self.name)

        if path:
            try:
                fig.savefig(path, dpi=500, format='png', bbox_inches='tight')
            except ValueError:
                print('Image is too large. Decreasing resolution of file to 100 DPI.')
                fig.savefig(path, dpi=100, format='png', bbox_inches='tight')


def get_colors(tree:Tree, data_path:str, taxonomy_data_path:str=None):
    ''''''
    # Create a dictionary mapping taxonomy to a list of selenoprotein contents
    taxonomy_data = pd.read_csv(taxonomy_data_path, delimiter='\t', usecols=['accession', 'gtdb_taxonomy']).rename(columns={'accession':'genome_id'})
    # Remove the RS_ and GB_ prefixes from the genome IDs so they can be matched to results.
    taxonomy_data['genome_id'] = taxonomy_data['genome_id'].apply(lambda x : re.sub('^[A-Z]+_', '', x))
    # No need to look at the entire taxonomy, so filter out those which match the root node of the tree. 
    taxonomy_data = taxonomy_data[taxonomy_data.gtdb_taxonomy.str.contains(f'{tree.rank_root}__{tree.name}')]

    # Read in the pickled DataFrame. 
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        data['genome_id'] = data.index # Make a genome ID column.
        data = data.reset_index()
        # Merge the results data and taxonomy data on the index, only keeping the genomes which are in the filtered taxonomy data. 
        data = data.merge(taxonomy_data, how='inner', on=['genome_id', 'gtdb_taxonomy'])

    colors = {leaf.name:[] for leaf in tree.get_terminals()}
    for row in tqdm(data[['hit_fraction', 'genome_id', 'gtdb_taxonomy']].itertuples(), total=len(data)):
        name = re.search(f'{tree.rank_leaves}__([\w\s]+)', row.gtdb_taxonomy).group(1)
        if name in colors:
            colors[name].append(row.hit_fraction)
    colors = {k:np.mean(v) if len(v) > 0 else 0 for k, v in colors.items()}

    def to_hex(r, g, b, a):
        '''Convert RGBA output tuple to a hex string.'''
        r, g, b = int(255 * r), int(255 * g), int(255 * b)
        return  '#{:02X}{:02X}{:02X}'.format(r, g, b)

    norm = mpl.colors.Normalize(vmin=min(list(colors.values())), vmax=max(list(colors.values())))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='Blues')
    colors = {k:to_hex(*cmap.to_rgba(v)) for k, v in colors.items()}

    return colors



if __name__ == '__main__':

    data_path = '/home/prichter/Documents/data/selenobot/results_11.12.2023.pkl'

    tree = read(ARCHAEA_TREE_PATH, 'newick')
    tree.collapse('p')
    print(tree.count_terminals(), f'leaf nodes present in the final tree.')
    colors = get_colors(tree, data_path=data_path, taxonomy_data_path=ARCHAEA_METADATA_PATH)
    tree.draw(path='tree.png', colors=colors) # , figsize=(12, 15))









