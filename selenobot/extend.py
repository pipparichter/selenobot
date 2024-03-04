'''Code for setting up databases for a search against GTDB using MMSeqs2.'''
import os
import pandas as pd
import numpy as np
import copy
from typing import NoReturn, List, Tuple, Dict
import re
from Bio.Seq import Seq


def check_nt_sequence(seq:str, reverse:bool=False, check_in_frame_stop:bool=False) -> bool:
    '''Checks to make sure a sequence of nucleotide bases has a valid start
    and stop codon. Assumes the reverse complement has already been found if the 
    sequence is on the reverse strand.'''
    if reverse:
        seq = str(Seq(seq).reverse_complement())

    assert len(seq) % 3 == 0, 'extend.check_nt_sequence: Sequence length is not divisible by three.'
    codons = [seq[3*i:3*i+3] for i in range(len(seq)//3)]

    assert codons[0] in ['ATG', 'GTG', 'TTG'], f'extend.check_nt_sequence: {codons[0]} is not a valid start codon.'
    assert codons[-1] in ['TAA', 'TAG', 'TGA'], f'extend.check_nt_sequence: {codons[-1]} is not a valid stop codon.'

    if check_in_frame_stop:
        for codon in codons[:-1]:
            assert codon not in ['TAA', 'TAG', 'TGA'], 'extend.check_nt_sequence: Sequence has an in-frame stop codon.' 


def extend_forward(nt_start, nt_stop, genome):
    '''Find a new nt_stop coordinate for a forward gene by reading until the next stop codon.'''
    # nt_stop coordinate is exclusive in a zero-indexed regime.  
    new_nt_stop = nt_stop
    while new_nt_stop < len(genome) - 3:
        codon = genome[new_nt_stop:new_nt_stop + 3]
        if codon in  ['TAA', 'TAG', 'TGA']:
            return nt_start, (new_nt_stop + 3)
        new_nt_stop += 3 # Shift the position by three (move to the next codon). 
    
    # Raise an exception if no other stop codon is found in the genome. 
    raise Exception('extend.extend_forward: No stop codon detected.')
    

def extend_reverse(nt_start, nt_stop, genome):
    '''Find a new nt_start coordinate for a reverse gene by reading until the next reversed nt_stop codon.'''
    new_nt_start = nt_start
    while new_nt_start >= 3:
        codon = genome[new_nt_start - 3:new_nt_start]
        if codon in ['TTA', 'CTA', 'TCA']:
            return new_nt_start - 3, nt_stop 
        new_nt_start -= 3 # Shift the start position to the left by three.

    # Raise an exception if no other stop codon is found in the genome. 
    raise Exception('extend.extend_reverse: No stop codon detected.')


def translate(nt_seq, reverse=False):
    '''Translate a nucleotide sequence, returning the corresponding sequence of amino acids.'''
    # It's possible that I will need to manually remove the stop codon if I set to_stop=True
    nt_seq = Seq(nt_seq)
    if reverse:
        nt_seq = nt_seq.reverse_complement()
    # Check the extended form of the nucleotide sequence. 
    check_nt_sequence(str(nt_seq))


    # The cds=True ensures that the sequence is in multiples of 3, and starts with a start codon. 
    aa_seq = nt_seq.translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  

    assert aa_seq[-1] == '*', 'extend.translate: The last symbol in the amino acid sequence should be *, indicating a translational stop.'
    aa_seq = aa_seq[:-1] # Remove the terminal * character. 

    aa_seq = str(aa_seq).replace('*', 'U') # Replace the first in-frame stop with a selenocysteine.

    assert '*' not in aa_seq, 'ecoli.translate: Too many in-frame stop codons in translated sequence.'
    return aa_seq

 
def extend(df:pd.DataFrame, genome:str):
    '''Take a DataFrame containing gene names and nt_start/nt_stop coordinates, which is read in from
    a GFF file. Then, find the gene in the input genome and extend it to the next stop codon. Add the 
    extended sequence to the DataFrame.'''
    # Need to do this, or else the original DataFrame is modified inplace. 
    df = df.copy()

    # Information to add to the DataFrame. 
    seqs = []
    nt_exts = []
    nt_starts = []
    nt_stops = []
    aa_lengths = []
    u_codons = [] 
    
    # removes = [] # Store IDs of sequences which are removed from the DataFrame. 

    for row in df.itertuples():

        # This code assumes the coding sequence frame starts at the first nucleotide. Confirm this is true. 
        assert int(row.frame) == 0, 'extend.extend: Expected reading frame to start at the first nucleotide position.'
        assert row.nt_start < row.nt_stop, 'extend.extend: Expected start to be to the left of the stop position.'

        # Check the original nucleotide sequence before extending.
        check_nt_sequence(genome[row.nt_start:row.nt_stop], reverse=row.reverse) # , check_in_frame_stop=True)

        # Handle forward and reverse genes differently. Reverse genes indicated by the complement flag. 
        nt_start, nt_stop = extend_forward(row.nt_start, row.nt_stop, genome) if not row.reverse else extend_reverse(row.nt_start, row.nt_stop, genome)
        nt_seq = genome[nt_start:nt_stop]

        # if check_nt_sequence(nt_seq):
        seqs.append(translate(nt_seq, reverse=row.reverse))

        u_codon = genome[row.nt_stop - 3:row.nt_stop] if not row.reverse else genome[row.nt_start:row.nt_start + 3]
        u_codon = u_codon if not row.reverse else str(Seq(u_codon).reverse_complement()) # Take the reverse complement if the sequence is on the reverse strand.
        u_codons.append(u_codon) # All codons should be as if they are read from the forward direction. 

        nt_exts.append((nt_stop - nt_start) - (row.nt_stop - row.nt_start)) # Store the amount the sequence was extended by. 
        nt_starts.append(nt_start)
        nt_stops.append(nt_stop)
        aa_lengths.append((nt_stop - nt_start) // 3) # Add the new length in terms of amino acids. 

    # Add new info to the DataFrame. 
    df['seq'] = seqs
    df['nt_ext'] = nt_exts
    df['nt_start'] = nt_starts
    df['nt_stop'] = nt_stops
    df['aa_length'] = aa_lengths
    df['u_codon'] = u_codons

    return df 



