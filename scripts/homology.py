'''Code for searching for homologs to extended predicted selenoproteins against the GTDB.'''
import pyopenms as oms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.Seq import Seq # Has a builtin method for coming up with the complement. 
import copy
from typing import NoReturn, List, Tuple, Dict
import re
import subprocess
from tqdm import tqdm

# Eventually will need to expand this list to accommodate all known selenos
known_selenoproteins = ['fdoG', 'fdnG', 'fdhF']
# From https://link.springer.com/article/10.3103/S0891416812030056#preview 
marker_genes = ['rpoB', 'gyrB', 'dnaK', 'dsrB', 'mipA', 'frc', 'oxc']


# In prokaryotes, E. coli is found to use AUG 83%, GUG 14%, and UUG 3% as START codons.

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


def load_results(dir_path=os.path.join(DATA_DIR, 'results_s_default')) -> pd.DataFrame:
    '''Load the BLAST hit results from the search of GTDB into a pandas DataFrame. BLAST result files (file extension m8)
    are tab-separated, with no column headers.'''
    # Columns in the output file, as specified in the MMSeqs2 User Guide. 
    cols = ['header', 'target_gene_id', 'seq_identity', 'alignment_length', 'num_mismatches', 'num_gap_openings', 'query_domain_start', 'query_domain_stop', 'target_domain_start', 'target_domain_stop', 'e_value', 'bit_score']

    def parse_headers(headers:pd.Series) -> pd.DataFrame:
        rows = []
        for header in headers:
            header = dict([item.split('=') for item in header.split('|')])
            # Don't need all of the information contained in the header... 
            header = {key:val for key, val in header.items() if key in ['gene_id', 'aa_length', 'nt_ext']}
            header['aa_ext'] = int(header['nt_ext']) // 3 # Get the number of added amino acids. 
            header = {'query_' + key:val for key, val in header.items()} # Clarify that all header information is associated with the query sequence.
            rows.append(header)
        return pd.DataFrame(rows, index=np.arange(len(headers)))

    data = []
    for filename in os.listdir(dir_path):
        # Grab the number of the split from the filename. 
        split = int(re.match('([0-9]+)', filename).group(1))
        split_data = pd.read_csv(os.path.join(dir_path, filename), delimiter='\t', names=cols)
        split_data = pd.concat([split_data.drop(columns=['header']), parse_headers(split_data.header)], axis=1)
        split_data['split'] = split # Add the target database split where the BLAST hit was found.
        data.append(split_data)

    data = pd.concat(data, axis=0)
    # Convert the columns which are numbers to numerical columns. Only the query and target gene_ids are not numerical
    num_cols = [col for col in data.columns if 'gene_id' not in col]
    data[num_cols] = data[num_cols].apply(pd.to_numeric)

    return data


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


def get_hits_past_stop_codon(data:pd.DataFrame, require_span:bool=True) -> pd.DataFrame: #, remove_query_gene_ids:List[str]=[]) -> pd.DataFrame:
    '''Find instances where a domain match is found past the first stop codon. The aa_length column gives the length of the
    extended amino acid sequence (if an extension was applied), so the stop codon location can be computed by subtracting the aa_ext
    from the length.
    
    :param data: The DataFrame containing the BLAST hits for the query sequences against GTDB.
    :param require_span: Whether or not to only include hits which span the U codon (not just in the region to the right of it).
    :param remove_query_gene_ids: If specified, the query hits to remove from the DataFrame (e.g. false positives like ilvB).
    :return: A DataFrame which contains the hits which meet the parameter specifications. 
    '''
    # Compute the location of the posited U residue. 
    data['query_u_position'] = data.query_aa_length - data.query_aa_ext
    # Filter by locations where the end of the matching domain in the query sequence is found past the extension.
    data = data[data.query_domain_stop >= data.query_u_position]
    if require_span: # Make sure the beginning of the matching domain is to the left of the putative U position. 
        data = data[data.query_domain_start <= data.query_u_position] 
    
    # Remove the desired query sequences/ 
    # data = data[~data.query_gene_id.isin(remove_query_gene_ids)]
    return data



# The goal is to find hits which are in the region after the stop codon of the predicted proteins, but don't overlap with the next protein. 
if __name__ == '__main__':

    results = load_results(dir_path=os.path.join(DATA_DIR, 'results_s_default'))
    controls = load_results(dir_path=os.path.join(DATA_DIR, 'controls_s_default'))
    # Remove the things we know to be false positives. 
    results = results[~results.query_gene_id.isin(['ilvB', 'ivbL'])]
    controls = controls[~controls.query_gene_id.isin(['ilvB', 'ivbL'])]

    print('Total hits:', len(results))
    print('Total hits (controls):', len(controls))

    # The results are the same whether or not we require spanning. 
    results = get_hits_past_stop_codon(results, require_span=False)
    print('Total hits spanning putative selenocysteine:', len(results))
    print('Query sequences with hits spanning putative selenocysteine:', set(results.query_gene_id))
    

