'''Code for setting up databases for a search against GTDB using MMSeqs2.'''
import os
import pandas as pd
import numpy as np
import copy
from typing import NoReturn, List, Tuple, Dict
import re


known_selenoproteins = ['fdoG', 'fdnG', 'fdhF']


def check_nt_sequence(seq:str) -> bool:
    '''Checks to make sure a sequence of nucleotide bases has a valid start
    and stop codon.'''

    def has_valid_start_codon(seq):
        '''Check if a nucleotide sequence has a valid start codon. Assumes reverse
        sequences have been flipped to standard direction.'''
        return seq[:3] in ['ATG', 'GTG', 'TTG']

    def has_valid_stop_codon(seq):
        '''Check if a nucleotide sequence has a valid stop codon. Assumes reverse
        sequences have been flipped to standard direction.'''
        return seq[-3:] in ['TAA', 'TAG', 'TGA']

    return has_valid_start_codon(seq) and has_valid_stop_codon(seq)


def extend_forward(nt_start, nt_stop, genome):
    '''Find a new nt_stop coordinate for a forward gene by reading until the next stop codon.'''
    # nt_stop coordinate seems to be the nucleotide right after the first stop. 
    new_nt_stop = nt_stop
    while new_nt_stop < len(genome) - 3:
        codon = genome[new_nt_stop:new_nt_stop + 3]
        if codon in  ['TAA', 'TAG', 'TGA']:
            return nt_start, (new_nt_stop + 3)
        new_nt_stop += 3 # Shift the position by three (move to the next codon). 
    
    return nt_start, nt_stop # Return the original start and stop if nothing is found. 
    

def extend_reverse(nt_start, nt_stop, genome):
    '''Find a new nt_start coordinate for a reverse gene by reading until the next reversed nt_stop codon.'''
    new_nt_start = nt_start
    while new_nt_start >= 3:
        codon = genome[new_nt_start - 3:new_nt_start]
        if codon in ['TTA', 'CTA', 'TCA']:
            return new_nt_start - 3, nt_stop 
        new_nt_start -= 3 # Shift the start position to the left by three

    return nt_start, nt_stop # Just return the original start and stop if nothing is found. 


def translate(nt_seq, reverse=False):
    '''Translate a nucleotide sequence, returning the corresponding sequence of amino acids.'''
    # It's possible that I will need to manually remove the stop codon if I set to_stop=True
    nt_seq = Seq(nt_seq)
    if reverse:
        nt_seq = nt_seq.reverse_complement()

    # The cds=True ensures that the sequence is in multiples of 3, and starts with a start codon. 
    aa_seq = nt_seq.translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  
    aa_seq = aa_seq[:-1] # Remove the asterisk termination character, becayse pyopenms doesn't like it. 
    aa_seq = str(aa_seq).replace('*', 'U', 1) # Replace the first in-frame stop with a selenocysteine. 
    assert '*' not in aa_seq, 'ecoli.translate: Too many in-frame stop codons in translated sequence.'
    return False, aa_seq

 
def extend(df:pd.DataFrame, genome:str):
    '''Take a DataFrame containing gene names and nt_start/nt_stop coordinates. Then, find the gene in the input
    genome and extend it to the next stop codon. Add the extended sequence to the DataFrame.'''
    seqs, nt_exts, nt_starts, nt_stops, aa_lengths = [], [], [], [], []
    removes = []

    for row in df.itertuples():
        # Don't extend known selenoproteins. 
        if row.gene_id not in known_selenoproteins: 
            # Handle forward and reverse genes differently. Reverse genes indicated by the complement flag. 
            nt_start, nt_stop = extend_forward(row.nt_start, row.nt_stop, genome) if not row.reverse else extend_reverse(row.nt_start, row.nt_stop, genome)
        else:
            nt_start, nt_stop = row.nt_start, row.nt_stop
        
        nt_seq = genome[nt_start:nt_stop]
        if check_nt_sequence(seq):
            aa_seq = translate(nt_seq, reverse=row.reverse)
        else: # If the sequence does not have a valid start and stop codon, skip over it. 
            removes.append(remove) # Whether or not the sequence was translated correctly. 
            seqs.append('') # Placeholder before removing. 
            print(f'extend.extend: Removing {row.gene_id}.')

        seqs.append(aa_seq) # Add the nt_extended sequence to the list. 
        nt_exts.append((nt_stop - nt_start) - (row.nt_stop - row.nt_start)) # Store the amount the sequence was extended by. 
        nt_starts.append(nt_start)
        nt_stops.append(nt_stop)
        aa_lengths.append((nt_stop - nt_start) // 3) # Add the new length in terms of amino acids. 

    # Add new info to the DataFrame. 
    df['seq'], df['nt_ext'], df['nt_start'], df['nt_stop'], df['aa_length'] = seqs, nt_exts, nt_starts, nt_stops, aa_lengths
    return data[~np.array(removes)] # Remove all sequences which were not translated. 




