'''Code for setting up databases for a search against GTDB using MMSeqs2.'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import NoReturn, List, Tuple, Dict
import re
import subprocess
from tqdm import tqdm
import seaborn as sns

# Eventually will need to expand this list to accommodate all known selenos
known_selenoproteins = ['fdoG', 'fdnG', 'fdhF']

DATA_DIR = '/home/prichter/Documents/data/selenobot/homology/'


def has_valid_start_codon(seq):
    '''Check if a nucleotide sequence has a valid start codon. Assumes reverse
    sequences have been flipped to standard direction.'''
    return seq[:3] in ['ATG', 'GTG', 'TTG']


def has_valid_stop_codon(seq):
    '''Check if a nucleotide sequence has a valid stop codon. Assumes reverse
    sequences have been flipped to standard direction.'''
    return seq[-3:] in ['TAA', 'TAG', 'TGA']


def extend_forward(nt_start, nt_stop, genome):
    '''Find a new nt_stop coordinate for a forward gene by reading until the next stop codon.'''
    # nt_stop coordinate seems to be the nucleotide right after the first stop. 
    new_nt_stop = nt_stop
    while new_nt_stop < len(genome) - 3:
        codon = genome[new_nt_stop:new_nt_stop + 3]
        if codon in  ['TAA', 'TAG', 'TGA']:
            return nt_start, (new_nt_stop + 3)
            # return genome[nt_start:new_nt_stop + 3], (new_stop + 3) - stop
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

    # assert len(nt_seq) % 3 == 0, 'ecoli.translate: Length of nucleotide sequence is not divisible by 3.'
    if len(nt_seq) % 3 != 0: # Returndomain_stop None if the nucleotide sequence is not divisible by 3. 
        return True, 'length not divisible by three.'
    if not has_valid_start_codon(nt_seq):
        return True, 'invalid START codon.'
    if not has_valid_stop_codon(nt_seq):
        return True, 'invalid STOP codon'

    # The cds=True ensures that the sequence is in multiples of 3, and starts with a start codon. 
    aa_seq = nt_seq.translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  
    aa_seq = aa_seq[:-1] # Remove the asterisk termination character, becayse pyopenms doesn't like it. 
    aa_seq = str(aa_seq).replace('*', 'U', 1) # Replace the first in-frame stop with a selenocysteine. 
    assert '*' not in aa_seq, 'ecoli.translate: Too many in-frame stop codons in translated sequence.'
    return False, aa_seq


# NOTE: Will need to have a special case for complementary genes. 
def get_sequences(data:pd.DataFrame, genome:str):
    '''Take a DataFrame containing, at minimum, gene names and nt_start/nt_stop coordinates. Then, find the gene in the input
    genome and nt_extend it to the next nt_stop codon. Add the extended sequence to the DataFrame.'''
    seqs, nt_exts, nt_starts, nt_stops, aa_lengths = [], [], [], [], []
    removes = []

    for row in data.itertuples():
        if row.extend: # Whether or not to extend the sequence past the stop codon. 
            # print(f'ecoli.get_sequences: Extending sequence {row.gene_id}.')
            # Handle forward and reverse genes differently. Reverse genes indicated by the complement flag. 
            nt_start, nt_stop = extend_forward(row.nt_start, row.nt_stop, genome) if not row.reverse else extend_reverse(row.nt_start, row.nt_stop, genome)
        else:
            nt_start, nt_stop = row.nt_start, row.nt_stop
        remove, seq = translate(genome[nt_start:nt_stop], reverse=row.reverse)
        removes.append(remove) # Whether or not the sequence was translated correctly. 
        if remove: print(f'ecoli.get_sequences: Removing {row.gene_id}: {seq}')

        seqs.append(seq) # Add the nt_extended sequence to the list. 
        nt_exts.append((nt_stop - nt_start) - (row.nt_stop - row.nt_start)) # Store the amount the sequence was extended by. 
        nt_starts.append(nt_start)
        nt_stops.append(nt_stop)
        aa_lengths.append((nt_stop - nt_start) // 3) # Add the new length in terms of amino acids. 

    # Add new info to the DataFrame. 
    data['seq'], data['nt_ext'], data['nt_start'], data['nt_stop'], data['aa_length'] = seqs, nt_exts, nt_starts, nt_stops, aa_lengths
    return data[~np.array(removes)] # Remove all sequences which were not translated. 



def load_genome(genome_id:str, path=None):
    '''Load in the complete nucleotide sequence of the genome.'''
    filename = genome_id + '.fasta'
    # path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        # lines = f.readlines()[1:] # Skip the header line. 
        lines = f.read().splitlines()[1:] # Skip the header line. 
        seq = ''.join(lines)
    return seq



def load_database(remove_decoys:bool=True) -> pd.DataFrame:
    '''Load the database.fasta file in as a pandas DataFrame.'''

    path = os.path.join(DATA_DIR, 'database.fasta')
    data = {'seq':[], 'gene_id':[], 'nt_ext':[], 'nt_stop':[], 'nt_start':[], 'aa_length':[], 'accession':[], 'reverse':[]}

    with open(path, 'r') as f:
        text = f.read()
        seqs = re.split(r'^>.*', text, flags=re.MULTILINE)[1:]
        # Strip all of the newline characters from the amino acid sequences. 
        seqs = [s.replace('\n', '') for s in seqs]
        headers = re.findall(r'^>.*', text, re.MULTILINE)

        for seq, header in zip(seqs, headers):
            # Headers are of the form |gene_id|col=value|...|col=value
            header = header.replace('>', '') # Remove the header marker. 
            header = header.split('|')
            for entry in header:
                col, value = entry.split('=')
                data[col].append(value)
            data['seq'].append(seq) # Add the sequence as well. 

    data = pd.DataFrame(data)
    # Convert to numerical datatypes. 
    data[['aa_length', 'nt_start', 'nt_stop', 'nt_ext']] = data[['aa_length', 'nt_start', 'nt_stop', 'nt_ext']].apply(pd.to_numeric)
    data['reverse'] = data['reverse'].apply(bool)

    if remove_decoys: # Remove decoys if this option is set. 
        data = data[~data.gene_id.str.contains('*', regex=False)]

    return data


def load_predictions() -> List[str]:
    '''Load in the gene IDs of the proteins predicted to be selenoproteins, excluding the known
    selenoproteins.'''
    path = os.path.join(DATA_DIR, 'predictions.csv')
    predictions = pd.read_csv(path).values.ravel()
    return list(predictions)



def load_coordinates(gene_ids:List[str]=None) -> pd.DataFrame:
    '''Load in the gene coordinates and other metadata as a pandas DataFrame, processing the start and
    stop locations as integers in two separate columns. This assumes the coordinate data was obtained from 
    the www.ncbi.nlm.nih.gov/datasets/gene/. 
    
    :return: A DataFrame with columns for start and stop locations, the RefSeq gene ID, and whether or not
        the gene is read in reverse. Also includes the length of the protein in amino acids for later
        validation. 
    '''
    path = os.path.join(DATA_DIR, 'coordinates.tsv')

    data = pd.read_csv(path, sep='\t', usecols=['Orientation', 'Begin', 'End', 'Symbol', 'Protein length', 'Protein accession'])
    data = data.rename(columns={'Orientation':'orientation', 'Begin':'nt_start', 'End':'nt_stop', 'Symbol':'gene_id', 'Protein length':'aa_length', 'Protein accession':'accession'}) # Rename the columns for my sanity. 
    data[['nt_start', 'nt_stop']] = data[['nt_start', 'nt_stop']].apply(pd.to_numeric) # Convert the start and stop to integers. 
    data['nt_start'] = data['nt_start'] - 1 # Shift all starts to be zero-indexed for my sanity...
    data['reverse'] = [o == 'minus' for o in data.orientation]
    data = data.drop(columns=['orientation'])
    data = data.fillna('none')

    if gene_ids is not None:
        # If a list of gene IDs is specified, filter out those IDs. 
        data = data[data.gene_id.isin(gene_ids)]

    return data


def database_write(data:pd.DataFrame, filename:str='database.fasta') -> NoReturn:
    '''Write the query database to a FASTA file.'''
    text = ''
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        for row in data.itertuples():
            text += f'>gene_id={row.gene_id}|nt_ext={row.nt_ext}|aa_length={row.aa_length}|nt_start={row.nt_start}|nt_stop={row.nt_stop}|accession={row.accession}|reverse={row.reverse}\n'

            # Split the sequence up into shorter, 60-character strings.
            n = len(row.seq)
            seq = [row.seq[i:min(n, i + 80)] for i in range(0, n, 80)]
            assert len(''.join(seq)) == n, 'homology.database_write: Part of the sequence was lost when splitting into lines.'
            text += '\n'.join(seq) + '\n'
        
        f.write(text)


# Eventually, will need to be able to support this for a whole list of genomes.
def database_build_query() -> NoReturn:
    '''Build a query data database, which contains the sequences to search for homology matches for.'''
    # Grab the coordinate information about the predicted selenoproteins only. Exclude known selenoproteins.
    database = load_coordinates(gene_ids=[g for g in load_predictions() if g not in known_selenoproteins])
    # Mark the sequences which will be extended past the first STOP codon. 
    # database['extend'] = [(gene_id not in known_selenoproteins) for gene_id in database.gene_id]
    database['extend'] = True 
    database = get_sequences(database, load_genome('', path=os.path.join(DATA_DIR, 'genome.fasta')))
    database_write(database, filename='query.fasta')


def database_build_control() -> NoReturn:
    '''Build a control query data database, which contains the non-extended selenoprotein sequences.'''
    # Grab the coordinate information about the predicted selenoproteins only. Exclude known selenoproteins.
    database = load_coordinates(gene_ids=[id_ for id_ in load_predictions() if id_ not in known_selenoproteins])
    database['extend'] = False # Don't extend anything here. 
    database = get_sequences(database, load_genome('', path=os.path.join(DATA_DIR, 'genome.fasta')))
    database_write(database, filename='control.fasta')

