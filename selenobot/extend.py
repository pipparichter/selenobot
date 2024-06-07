'''Code for setting up databases for a search against GTDB using MMSeqs2.'''
import os
import pandas as pd
import numpy as np
import copy
from typing import NoReturn, List, Tuple, Dict
import re
from Bio.Seq import Seq

# TODO: Put more thought into the "orientation" column name. Would strand be better or more clear? What is convention?

START_CODONS = {'+':['ATG', 'GTG', 'TTG'], '-':['CAT', 'CAC', 'CAA']}
STOP_CODONS = {'+':['TAA', 'TAG', 'TGA'], '-':['TTA', 'CTA', 'TCA']}

def get_reverse_complement(seq:str) -> str:
    '''Converts the input DNA strand to its reverse complement.
    
    :param seq
    :param str
    '''
    seq = Seq(seq)
    seq = seq.reverse_complement()
    return str(seq)

def get_start_codon(seq:str, orientation:str='+') -> str:
    if orientation == '+':
        return seq[:3]
    elif orientation == '-':
        return seq[-3:]

def get_stop_codon(seq:str, orientation:str='+') -> str:
    if orientation == '-':
        return seq[:3]
    elif orientation == '+':
        return seq[-3:]


def get_codons(seq:str, orientation:str='+') -> List[str]:
    offset = len(seq) % 3 # Why did I allow for sequences with lengths that are not divisible by 3?
    if (orientation == '+') and (offset > 0):
        seq = seq[:-offset]
    elif (orientation == '-') and (offset > 0):
        seq = seq[offset:]
    assert len(seq) % 3 == 0, 'get_codons: Something went wrong when truncating the sequence.'
    seq = list(seq) 
    codons = np.array_split(seq, len(seq) // 3)
    codons = [''.join(list(codon)) for codon in codons]
    return codons


def check_start_codon(seq:str, orientation:str='+'):
    '''Check a gene to make sure it has a valid start codon.'''
    assert len(seq) % 3 == 0, 'is_valid_gene: Length of nucleotide sequence is not divisible by three.'
    start_codon = get_start_codon(seq, orientation=orientation)
    assert start_codon in START_CODONS[orientation], f'is_valid_gene: {start_codon} is not a valid start codon. Orientation is {orientation}.'


def check_stop_codon(seq:str, orientation:str='+'):
    '''Check a gene to make sure it has a valid stop codon.'''
    assert len(seq) % 3 == 0, 'is_valid_gene: Length of nucleotide sequence is not divisible by three.'
    stop_codon = get_stop_codon(seq, orientation=orientation)
    assert start_codon in START_CODONS[orientation], f'is_valid_gene: {start_codon} is not a valid start codon. Orientation is {orientation}.'


# NOTE: Does the range include the stop codon?
def extend(start:int=None, stop:int=None, contig:str=None, orientation:str='+', verbose:bool=False, **kwargs) -> Dict:
    '''Scan the DNA strand until the next stop codon is encountered.    

    :param start: The starting location of the gene in the sequence. Assumed to be inclusive.
    :param stop: The ending location of the gene in the sequence. Assumed to be inclusive. 
    :param contig: The contig on which the gene is found. 
    :param orientation: Either '+' or '-', indicating whether the gene is found on the forward or reverse strand, respectively.
    :return: A dictionary containing information about the extension. 
    '''
    # To account for the assumption that the upper bound is inclusive. 
    start -= 1

    gene = contig[start:stop]
    results = {'seq':gene}

    check_start_codon(gene, orientation=orientation) # Make sure the gene has a start codon. 

    original_length, original_stop_codon = len(gene), get_stop_codon(gene, orientation=orientation)

    if orientation == '+':
        step = 1
        # Get the codons from the original stop location (the end of the stop codon) to the end of the sequence. 
        codons = get_codons(contig[stop:], orientation=orientation)
    if orientation == '-':
        step = -1
        # Get the codons from the beginning of the contig to right before the stop codon. 
        codons = get_codons(contig[:start], orientation=orientation)
        original_stop_codon = get_reverse_complement(original_stop_codon)

    for codon in codons[::step]:
        if orientation == '-':
            gene = codon + gene
        elif orientation == '+':
            gene = gene + codon
        if codon in STOP_CODONS[orientation]:
            if verbose: print(f'extend: Detected stop codon {codon}.')
            break
    
    if not (codon in STOP_CODONS[orientation]):
        return None
    else:
        nt_ext = len(gene) - original_length # Get the size of the extension. 
        aa_ext = nt_ext // 3 # Get the extension size in amino acids. 
        
        if verbose: print(f'extend: Gene extended by {nt_ext} nucleotides.')
        return {'seq_ext':gene, 'original_stop_codon':original_stop_codon, 'nt_ext':nt_ext, 'aa_ext':aa_ext}

df_ext = []
for id_, id_df in df.groupby('id'): # Each id_df should only have one row. 
    row = id_df.as_dict(orient='records')[0]
    scaffold_id = id_df.scaffoldId.item()
    contig = genome_df[genome_df.scaffold_id.str.fullmatch(scaffold_id)].seq.item()
    row.update(extend(contig=contig, **row))
    extensions_df.append(row)

extensions_df = pd.DataFrame(extensions_df)
extensions_df['seq'] = extensions_df.apply(lambda row : translate(row['seq'], orientation=row['orientation']), axis=1)


def translate(seq, orientation:str='+'):
    '''Translate a nucleotide sequence.'''

    check_gene(seq, orientation=orientation)
    # Convert the sequence to its reverse complement if the orientation is '-'.
    seq = get_reverse_complement(seq) if orientation == '-' else seq
    # Translate the sequence using the BioPython module. 
    seq = Seq(seq).translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  

    assert seq[-1] == '*', 'extend.translate: The last symbol in the amino acid sequence should be *, indicating a translational stop.'
    seq = seq[:-1] # Remove the terminal * character. 

    seq = str(seq).replace('*', 'U') # Replace the first in-frame stop with a selenocysteine.
    seq = 'M' + seq[1:] # Not sure why this is always methionine in NCBI, but doesn't always get translated as M. 

    assert '*' not in seq, 'translate: Too many in-frame stop codons in translated sequence.'
    return seq


# def extend(df:pd.DataFrame, genome:str):
#     '''Take a DataFrame containing gene names and nt_start/nt_stop coordinates, which is read in from
#     a GFF file. Then, find the gene in the input genome and extend it to the next stop codon. Add the 
#     extended sequence to the DataFrame.'''
#     # Need to do this, or else the original DataFrame is modified inplace. 
#     df = df.copy()

#     # Information to add to the DataFrame. 
#     seqs = []
#     nt_exts = []
#     nt_starts = []
#     nt_stops = []
#     aa_lengths = []
#     u_codons = [] 
    
#     # removes = [] # Store IDs of sequences which are removed from the DataFrame. 

#     for row in df.itertuples():

#         # This code assumes the coding sequence frame starts at the first nucleotide. Confirm this is true. 
#         assert int(row.frame) == 0, 'extend.extend: Expected reading frame to start at the first nucleotide position.'
#         assert row.nt_start < row.nt_stop, 'extend.extend: Expected start to be to the left of the stop position.'

#         # Check the original nucleotide sequence before extending.
#         check_nt_sequence(genome[row.nt_start:row.nt_stop], reverse=row.reverse) # , check_in_frame_stop=True)

#         # Handle forward and reverse genes differently. Reverse genes indicated by the complement flag. 
#         nt_start, nt_stop = extend_forward(row.nt_start, row.nt_stop, genome) if not row.reverse else extend_reverse(row.nt_start, row.nt_stop, genome)
#         nt_seq = genome[nt_start:nt_stop]

#         # if check_nt_sequence(nt_seq):
#         seqs.append(translate(nt_seq, reverse=row.reverse))

#         u_codon = genome[row.nt_stop - 3:row.nt_stop] if not row.reverse else genome[row.nt_start:row.nt_start + 3]
#         u_codon = u_codon if not row.reverse else str(Seq(u_codon).reverse_complement()) # Take the reverse complement if the sequence is on the reverse strand.
#         u_codons.append(u_codon) # All codons should be as if they are read from the forward direction. 

#         nt_exts.append((nt_stop - nt_start) - (row.nt_stop - row.nt_start)) # Store the amount the sequence was extended by. 
#         nt_starts.append(nt_start)
#         nt_stops.append(nt_stop)
#         aa_lengths.append((nt_stop - nt_start) // 3) # Add the new length in terms of amino acids. 

#     # Add new info to the DataFrame. 
#     df['seq'] = seqs
#     df['nt_ext'] = nt_exts
#     df['nt_start'] = nt_starts
#     df['nt_stop'] = nt_stops
#     df['aa_length'] = aa_lengths
#     df['u_codon'] = u_codons

#     return df 



