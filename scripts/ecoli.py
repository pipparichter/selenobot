'''Script for validating results against an E. Coli genome.'''
import pyopenms as oms
import os
import pandas as pd
import numpy as np
from pyopenms.plotting import plot_spectrum # Takes an MSSpectrum as input. 
import matplotlib.pyplot as plt
from Bio.Seq import Seq # Has a builtin method for coming up with the complement. 
from selenobot.utils import pd_from_fasta, pd_to_fasta, fasta_seqs
# from Bio.Alphabet import generic_dna
import copy
from typing import NoReturn, List
import re
import subprocess

# NOTE: The genome file is one-indexed, so the start position must be shifted backwards by one. 
# NOTE: For reverse genes, it seems as though I need to grab the *reverse* complement of the input nucleotide sequence. 


# Directory where the mzML files are stored. 
MZML_PATH = '/home/prichter/Documents/data/selenobot/ecoli/ms/mzml/'
ECOLI_PATH = '/home/prichter/Documents/data/selenobot/ecoli/'

FORWARD_STOP_CODONS = ['TAA', 'TAG', 'TGA']
REVERSE_STOP_CODONS = ['TTA', 'CTA', 'TCA']
# In prokaryotes, E. coli is found to use AUG 83%, GUG 14%, and UUG 3% as START codons.
FORWARD_START_CODONS = ['ATG', 'GTG', 'TTG']
REVERSE_START_CODONS = ['CAT', 'CAC', 'CAA']

# Known selenoproteins in E. coli, which we do not want to extend. All are formate dehydrogenase-related. 
# KNOWN_SELENOPROTEINS = ['eco:b1474', 'eco:b3894', 'eco:b4079']
KNOWN_SELENOPROTEINS = ['fdoG', 'fdnG', 'fdhF'] # Use the RefSeq IDs. 


def has_valid_start_codon(seq, reverse=False):
    '''Check if a nucleotide sequence has a valid start codon.'''
    if reverse:
        return seq[-3:] in REVERSE_START_CODONS
    else:
        return seq[:3] in FORWARD_START_CODONS


def has_valid_stop_codon(seq, reverse=False):
    '''Check if a nucleotide sequence has a valid stop codon.'''
    if reverse:
        return seq[:3] in REVERSE_STOP_CODONS
    else:
        return seq[-3:] in FORWARD_STOP_CODONS


def check_valid_sequence(seq):
    '''Check to confirm if a sequence translated from nucleotides in the genome is present in
    the set of genes in the FASTA file with E. coli genes.'''
    print(seq)
    valid_seqs = fasta_seqs(os.path.join(ECOLI_PATH, 'sequences.fasta'))
    assert seq in valid_seqs, 'ecoli.check_valid_sequence: The input sequence is not present in the E. coli proteome.'


def check_valid_start_stop_codons(data):
    '''Scan all nucleotide sequences in the input DataFrame and confirm each has a valid stop and
    start codon. Because of in-frame stop codons following extension, BioPython cannot be used for this.
    '''
    def check(seq, gene_id, reverse=False):
        valid_start = has_valid_start_codon(seq, reverse=reverse)
        valid_stop = has_valid_stop_codon(seq, reverse=reverse)
        assert valid_start, f'ecoli.check_start_stop_codons: {gene_id} has an invalid start codon.'
        assert valid_stop, f'ecoli.check_start_stop_codons: {gene_id} has an invalid stop codon.'

    data.apply(lambda row : check(row['nt_seq'], row['gene_id'], reverse=row['complement']), axis=1)


def filter_valid_start_stop_codons(data, filter_start=True, filter_stop=True):
    '''Scan all nucleotide sequences in the input DataFrame and remove those which do not have a valid start and
    stop codon. I have not yet figured out why this is.'''
    if filter_start:
        idxs = data.apply(lambda row : has_valid_start_codon(row['nt_seq'], reverse=row['complement']), axis=1)
        print(f'ecoli.filter_start_stop_codons: Detected {sum(~idxs.values)} sequences with invalid start codons, and removed them from the DataFrame.')
        data = data[idxs]
    if filter_stop:
        idxs = data.apply(lambda row : has_valid_stop_codon(row['nt_seq'], reverse=row['complement']), axis=1)
        print(f'ecoli.filter_start_stop_codons: Detected {sum(~idxs.values)} sequences with invalid stop codons, and removed them from the DataFrame.')
        data = data[idxs]

    return data


def extend_forward(start, stop, genome):
    '''Find a new stop coordinate for a forward gene by reading until the next stop codon.'''
    # Stop coordinate seems to be the nucleotide right after the first stop. 
    new_stop = stop
    while new_stop < len(genome) - 3:
        codon = genome[new_stop:new_stop + 3]
        if codon in FORWARD_STOP_CODONS:
            return start, (new_stop + 3)
            # return genome[start:new_stop + 3], (new_stop + 3) - stop
        new_stop += 3 # Shift the position by three (move to the next codon). 
    
    return start, stop # Return the original start and stop if nothing is found. 
    

def extend_reverse(start, stop, genome):
    '''Find a new start coordinate for a reverse gene by reading until the next reversed stop codon.'''
    new_start = start
    while new_start >= 3:
        codon = genome[new_start - 3:new_start]
        if codon in REVERSE_STOP_CODONS:
            if codon == 'TAA':
                print(codon)
            return new_start - 3, stop 
        new_start -= 3 # Shift the start position to the left by three

    return start, stop # Just return the original start and stop if nothing is found. 


def translate(nt_seq, reverse=False):
    '''Translate a nucleotide sequence, returning the corresponding sequence of amino acids.'''
    # It's possible that I will need to manually remove the stop codon if I set to_stop=True
    nt_seq = Seq(nt_seq)
    assert len(nt_seq) % 3 == 0, 'ecoli.get_amino_acid_sequences: Length of nucleotide sequence is not divisible by 3.'
    if reverse:
        nt_seq = nt_seq.reverse_complement()
    # The cds=True ensures that the sequence is in multiples of 3, and starts with a start codon. 
    aa_seq = nt_seq.translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  
    aa_seq = str(aa_seq).replace('*', 'U', 1) # Replace the first in-frame stop with a selenocysteine. 
    return aa_seq[:-1] # Also remove the asterisk termination character, becayse pyopenms doesn't like it. 


# NOTE: Will need to have a special case for complementary genes. 
def get_extended_sequences(data:pd.DataFrame, genome:str):
    '''Take a DataFrame containing, at minimum, gene names and start/stop coordinates. Then, find the gene in the input
    genome and extend it to the next stop codon. Add the extended sequence to the DataFrame.'''
    seqs, exts, starts, stops = [], [], [], []
    for row in data.itertuples():

        # Handle forward and reverse genes differently. Reverse genes indicated by the complement flag. 
        start, stop = extend_forward(row.start, row.stop, genome) if not row.complement else extend_reverse(row.start, row.stop, genome)
        seq = translate(genome[start:stop], reverse=row.complement)

        seqs.append(seq) # Add the extended sequence to the list. 
        exts.append((stop - start) - (row.stop - row.stop)) # Store the amount the sequence was extended by. 
        starts.append(start)
        stops.append(stop)

    # Add new info to the DataFrame. 
    data['seq'], data['ext'], data['start'], data['stop'] = seqs, exts, starts, stops
    return data


def get_decoy_sequences(data:pd.DataFrame) -> pd.DataFrame: 
    '''Generates decoy protein sequences by reversing every original sequence. Returns a DataFrame with both 
    the original sequences and the original sequences.'''

    decoy_data = copy.deepcopy(data)
    # Reverse each sequence in the database. 
    decoy_data['seq'] = data.seq.apply(lambda s : s[::-1])
    # Label all decoys with an asterisk. 
    decoy_data['gene_id'] = data.gene_id.apply(lambda i : i + '*')

    return pd.concat([data, decoy_data])


def load_mzml(filename):
    mzml = oms.MSExperiment()
    oms.MzMLFile().load(os.path.join(MZML_PATH, 'A14-07017.mzML'), mzml)
    return mzml


def load_sequences():
    '''Load the FASTA file containing the E. coli gene sequences. .'''
    df = pd_from_fasta(os.path.join(ECOLI_PATH, 'sequences.fasta'), get_id_func=lambda h : re.search('gene=([a-zA-Z]+)', h).group(1))
    df['gene_id'] = df.id
    return df.drop(columns=['id'])


def load_genome():
    '''Load in the complete nucleotide sequence of the genome.'''
    with open(os.path.join(ECOLI_PATH, 'genome.fasta'), 'r') as f:
        # lines = f.readlines()[1:] # Skip the header line. 
        lines = f.read().splitlines()[1:] # Skip the header line. 
        seq = ''.join(lines)
    return seq


def load_predictions(path:str=os.path.join(ECOLI_PATH, 'predictions.csv')) -> List[str]:
    '''Load in the gene IDs of the proteins predicted to be selenoproteins, excluding the known
    selenoproteins.'''
    predictions = pd.read_csv(path, usecols=['info'])
    # Exclude the known selenoproteins from the list. 
    return [re.search('\(RefSeq\) ([A-Za-z]+);', i).group(1) for i in predictions['info'] if i not in KNOWN_SELENOPROTEINS]


def load_coordinate_data(path:str=os.path.join(ECOLI_PATH, 'coordinates.tsv')) -> pd.DataFrame:
    '''Load in the gene coordinates and other metadata as a pandas DataFrame, processing the start and
    stop locations as integers in two separate columns.
    
    :param path: The path to the coordinate data. 
    :return: A DataFrame with columns for start and stop locations and the RefSeq gene ID. 
    '''

    data = pd.read_csv(path, sep='\t', names=['gene_id', 'type', 'coords', 'description'])
    data['gene_id'] = data.description.apply(lambda d : d.split(';')[0]) # Replace the current gene IDs with the RefSeq IDs. 
    data = data[data['type'].str.match('CDS')] # Filter out everything that is not a coding sequence.

    # Flag the genes on reverse strands. 
    data['complement'] = ['complement' in c for c in data['coords']]
    # Remove coordinates with "joins" (none of the predicted selenoproteins have this)
    data = data[~data['coords'].str.contains('join')] 
    # Remove complementary(...) from the coords which have it. 
    data['coords'] = [c.replace('complement(', '').replace(')', '') for c in data['coords']] 
    data[['start', 'stop']] = data['coords'].str.split('\.\.', expand=True) # Make sure to escape the dot character so it doesn't think it's a pattern. 
    data[['start', 'stop']] = data[['start', 'stop']].apply(pd.to_numeric) # Convert start and stop to integers. 
    # Shift all starts to be zero-indexed for my sanity...
    data['start'] = data['start'] - 1

    return data.drop(columns=['description', 'type', 'coords']) # Drop unnecessary columns. 


def run_tandem(path:str=os.path.join(ECOLI_PATH, 'ms/mgf/')) -> NoReturn:
    '''Run X! Tandem on a directory of mgf files.'''
    for sample in os.listdir(path):
        sample = sample.replace('.mgf', '') # Remove the file extension. 
        # Read in the input.xml template file, and fill in the sample name. 
        with open('input.xml', 'r') as f:
            input_xml = f.read().format(sample=sample)
        with open('tmp.xml', 'w') as f:
            f.write(input_xml)
        
        cmd = '/home/prichter/tandem-linux-17-02-01-4/bin/tandem.exe'
        subprocess.run([cmd, 'tmp.xml'])
        # Remove the temporary input.xml file. 
        subprocess.run(['rm', 'tmp.xml'])



def build_fasta_database(path=os.path.join(ECOLI_PATH, 'search.fasta')) -> NoReturn:
    
    genome = load_genome()

    # Grab the coordingate information about the predicted selenoproteins only. 
    data = load_coordinate_data()
    data = data[data['gene_id'].isin(load_predictions())]
    data = get_extended_sequences(data, genome)

    # Combine the extended sequences with the original data. 
    data = pd.concat([load_sequences(), data[['seq', 'gene_id']]])
    # Add the decoy sequences. 
    data = get_decoy_sequences(data)

    pd_to_fasta(data.set_index('gene_id'), path)


if __name__ == '__main__':
    
    # build_fasta_database()
    run_tandem()



# eco:b3894 seems to have an extra in-frame stop codon which I don't quite understand. It also has a U present, which means
# that this is not the issue.
# The issue seems to be that there is already a stop codon in the gene boundaries set in the file (i.e. already known selenoproteins).
# Will need to add a special case for this.  