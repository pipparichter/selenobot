'''Utilities for visualizing phylogenetic trees with selenoprotein data.'''

import ete3
import PyQt5
from ete3 import NodeStyle
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# from PySide6.QtWidgets import QGraphicsItem


from typing import NoReturn, Tuple, List, Dict

# Tree plotting works with the GTDB data, which should contain the following information:
# - Parsed taxonomy information (domain, class, phylum, etc.)
# - total_hits, i.e. detected selenoproteins. 
# - selenoprotein_ratio, the ratio of total_hits to the number of genes. 
# - selD_copy_num, number of selD copies detected using Kofamscan
# - total_genes, total number of genes in the genome (I think detected using Prodigam)
# - hits_with_annotation, total number of genes detected as selenoproteins, successfully annotated (I think using Kofamscan?)
# - total_genes_with_annotation, total number of genes which are successfully annotated. 
# - genome_id, the unique genome identifier. 

LEVELS = ['domain', 'phylum', 'class', 'order', 'genus', 'species']


# TODO: Something that displayes species name when hovering!
class InteractiveNode(QGraphicsItem):
    pass


def load_tree(tree_path:str, collapse=True) -> ete3.Tree:
    '''Loads an ete3 tree in Newick format from the specified path.'''

    # Read in the tree data from the file. Should be in paranthetical format. 
    with open(tree_path, 'r') as f:
        tree_data = f.read()

    # By default, the species naming function just grabs the first three letters of the
    # node name (which is like d__ or s__). 
    def sp_naming_function(name):
        if 's__' in name:
            return name.split('s__')[-1]
        else:
            return ''

    t = ete3.PhyloTree(tree_data, format=1, quoted_node_names=True, sp_naming_function=sp_naming_function) 
    if collapse:
        t = t.collapse_lineage_specific_expansions(return_copy=True)
    return t


def rgb_to_hex(r, g, b):
    '''Convert a red, blue, and green value (the output of map.to_rgba) to a hex string.'''
    # First need to put each value into hexadecimal. 
    return '#' + '%02x%02x%02x' % (r, g, b)



def parse_taxonomy(taxonomy:str) -> Dict[str, str]:
    '''Extract information from a taxonomy string.'''

    m = {'o':'order', 'd':'domain', 'p':'phylum', 'c':'class', 'f':'family', 'g':'genus', 's':'species'}

    parsed_taxonomy = {}
    # Split taxonomy string along the semicolon...
    for x in taxonomy.strip().split(';'):
        l, t = x.split('__')
        l = l.strip() # Make sure there's no remaining whitespace. 
        t = t.strip()
        parsed_taxonomy[m[l]] = t
    
    return parsed_taxonomy # Return None if flag is not found in the taxonomy string. 

    
# def get_average_selenoprotein_ratio(level:str, name:str, gtdb_data:pd.DataFrame=None) -> float:
#     '''Calculate the average selenoprotein_ratio across all members of a taxonomic level specified
#     by the given name.'''
#     # Search the DataFrame for wherever the taxonomical level is equal to the given name. 
#     data = gtdb_data[gtdb_data[level] == name]
#     return selenoprotein_ratios = data['selenoprotein_ratio'].mean()


def label_nodes_with_taxonomy(tree:ete3.Tree, gtdb_data=None) -> NoReturn:
    '''
    Traverses the tree, and labels all nodes with relevant taxonomical information. 
    '''
    # Should only need to annotate the leaf nodes to filter by taxonomy. Although
    # how do we find the root node?

    for leaf in tree.get_leaves(): 
        # family = genome_id_to_family_map[node.name]
        entry = gtdb_data[gtdb_data['species'] == leaf.species]

        if len(entry) == 0:
            print(f'tree.label_nodes_`with_taxonomy: No entries matching {leaf.species} found in gtdb_data.')
            continue
        if len(entry) > 1:
            print(f'tree.label_nodes_with_taxonomy: More than one entry matching {leaf.species} found in gtdb_data.')
            continue
        
        leaf.add_features(**{level:entry[level].item() for level in LEVELS})


def get_is_leaf_fn(level:str):
    '''Generates a leaf function for the specified level.'''

    def is_leaf_fn(node):
        '''Function for collapsing the tree according to taxonomy.'''
        # If a node is monophyletic for the taxonomy, then it is a leaf.
        leaves = node.get_leaves()
        # Get the taxonomy of the organism at the specified level.
        x = getattr(leaves[0], level)
        
        if np.all([x == getattr(leaf, level) for leaf in leaves]):
            node.name = x
            return True
        else:
            return False
    
    return is_leaf_fn # Return the function defined using the input level.


def find_root_node(tree:ete3.Tree, level:str, name:str) -> ete3.Tree:
    '''Traverses the phylogenetic tree, finding the node for which all
    children belong to the phylogenetic category specified in the input.'''
    for node in tree.traverse:
        leaves = node.get_leaves()
        taxonomies = np.array([getattr(leaf, level) for leaf in leaves])
        if np.all(taxonomies == name):
            return node 
    
    # Raise an exception if no node was found.     
    raise Exception(f'tree.find_root_node: No root node {name} was found at taxonomical level {level}.')


# def merge_nodes_by_family(tree:ete3.Tree, families=List[str], genome_id_to_family_map:Dict[str,str]=None) -> ete3.Tree :
def merge_nodes_by_taxonomy(tree:ete3.Tree, gtdb_data=None, level:str=None) -> ete3.Tree :
    '''Takes a tree as input, and merges all the nodes so that there is only one node for
    a particular family.'''

    # Annotate all nodes with stored taxonomical data. 
    label_nodes_with_taxonomy(tree, gtdb_data=gtdb_data)
    return ete3.Tree(tree.write(is_leaf_fn=get_is_leaf_fn(level)), format=1, quoted_node_names=True)


# TODO: Want to be able to specify the root node of a plot, as well as the level. 
# Might be worth moving all the tree plotting code over to another Python file. 

# Having an issue where not every organism in the tree is present in the GTDB data. 

def plot(
        tree:ete3.Tree,
        gtdb_data:pd.DataFrame, 
        level:str='class',
        sel_ratio_threshold:float=None,
        path:str=None, 
        show_sel_ratio:bool=True,
        font_size:int=15, 
        margin:int=10,
        root:str=None, 
        root_level:str='domain') -> NoReturn:
    '''Phylogenetic tree from the GTDB database with selenoprotein content coded as a color. 
    Selenoprotein content is given as total selenoproteins divided by total genes.'''

    assert path is not None, 'tree.plot: A path must be specified.'

    os.environ['QT_QPA_PLATFORM'] = 'wayland'

    # The level of the root node must be lower than that of the leaves. 
    if root is not None:
        assert root_level is not None, 'tree.plot: A root level must be specified along with a node to use as the root.'
        assert levels.index(root_level) < levels.index(level), f'tree.plot: Root level {root_level} must be higher than leaf level {level}.'
        # Make sure the root name is in the correct level. 
        assert root in np.unique(gtdb_data[[level]].values), f'tree.plot: Root name is not in the specified taxonomic level {root_level}.'


    print(f'tree.plot_gtdb_tree: {len(np.unique(gtdb_data[level].values))} unique instances of level "{level}" present in gtdb_data.')

    tree = merge_nodes_by_taxonomy(tree, gtdb_data=gtdb_data, level=level)

    selenoprotein_ratios = gtdb_data.groupby(level)['selenoprotein_ratio'].mean()

    # Create a normalized colormap for coloring the nodes. 
    cmap = matplotlib.colormaps['Blues']
    norm = matplotlib.colors.Normalize(vmin=min(selenoprotein_ratios), vmax=max(selenoprotein_ratios))
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

            sel_ratio = selenoprotein_ratios[selenoprotein_ratios.index == node.name].item()

            # If a sec_content threshold is specified, remove all leaves which do not meet the threshold.
            if sel_ratio_threshold is not None:
                if selenoprotein_ratio < sel_ratio:
                    node.delete()
                    continue

            label = node.name + f' ({sel_ratio})' if show_sel_ratio else node.name

            r, g, b, _ = m.to_rgba(sel_ratio, bytes=True) # Set bytes=True to get values between 0 and 255. 
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

    print(f'tree.plot_gtdb_tree: {len(tree.get_leaves())} leaves in final tree.')

    tree.render(path, tree_style=tree_style)



# def get_taxonomy_to_sec_content_map(gtdb_data, genome_id_to_taxonomy_map:Dict[str, str]=None):
#     '''Use GTDB taxomonomy to combine different families and plot average selenoprotein content
#     across labeled members.

#     '''
#     assert genome_id_to_taxonomy_map is not None, 'tree.get_class_to_sec_content: A dictionary mapping genome ID to class name must be specified.'
    
#     # Get a list of all possible taxonomical categories at the specified level. 
#     keys = np.unique([t for t in  genome_id_to_taxonomy_map.values()])
#     taxonomy_to_sec_content_map = {t:[] for t in keys}

#     for row in gtdb_data.itertuples():
#         # Get the relevant taxonomy. 
#         try: # Try to extract the specified taxonomic label. 
#             taxonomy = genome_id_to_taxonomy_map.get(row.Index, None)     
#             sec_content = row.total_hits / row.total_genes
#             taxonomy_to_sec_content_map[taxonomy].append(sec_content)
#         except KeyError: # If the taxonomy is not found, move on to the next iteration. 
#             continue

#     # Take the average of each sec content list. 
#     # Possibly will need to make sure that we are not trying to take the mean of an empty list.
#     return {t:0 if np.isnan(np.mean(s)) else np.mean(s) for t, s in taxonomy_to_sec_content_map.items()}


# def get_genome_id_to_taxonomy_map(taxonomy_data:pd.DataFrame, level:str='class') -> Dict[str, str]:
#     '''Processes a taxonomy file, converting it into a dictionary mapping genome ID
#     to the name of the family to which the genome belongs.

#     args:
#         - taxonomy_data: Contains two columns, genome_id and taxonomy. The taxonomy column contains
#             strings with each taxonomical level separated by semicolons. 
#         - level: The taxonomic level for which to accumulate data. 
#     '''
#     # Collect information into a dictionary mapping 
#     genome_id_to_taxonomy_map = {}
#     for row in taxonomy_data.itertuples():
#         try:
#             genome_id_to_taxonomy_map[row.genome_id] = parse_taxonomy(row.taxonomy)[level]
#         except KeyError:
#             print(f'tree.get_genome_id_to_taxonomy_map: Genome {genome_id} does not have specified taxonomy data.')
#             continue

#     return genome_id_to_taxonomy_map

# def label_nodes_with_taxonomy(tree:ete3.Tree, genome_id_to_taxonomy_map:Dict[str, str]=None) -> NoReturn:
#     '''Label all nodes in a tree with their family. Labels according to the specified taxonomic level.
    
#     args:
#         - tree: A Tree object with unlabeled nodes. 
#         - genome_id_to_taxonomy_map: A map co
#     '''
    
#     assert genome_id_to_taxonomy_map is not None, 'tree.get_class_to_sec_content: A dictionary mapping genome ID to class name must be specified.'

#     for leaf in tree.get_leaves():
#         # family = genome_id_to_family_map[node.name]
#         taxonomy = genome_id_to_taxonomy_map[leaf.name]
#         leaf.add_features(taxonomy=taxonomy)

# def get_support(name):
#     '''Extract support value from a node name. I feel like this should be done automatically?'''
#     if ':' in name:
#         support, name = name.split(':')
#         return float(support), name
#     else:
#         return None, name


