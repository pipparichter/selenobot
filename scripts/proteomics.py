'''Script for validating results against an E. Coli genome.'''
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
import xml
from tqdm import tqdm


known_selenoproteins = ['fdoG', 'fdnG', 'fdhF']

# In prokaryotes, E. coli is found to use AUG 83%, GUG 14%, and UUG 3% as START codons.

def get_ecoli_dir_path(strain):
    return f'/home/prichter/Documents/data/selenobot/ecoli/{strain}/'


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


def get_decoy_sequences(data:pd.DataFrame) -> pd.DataFrame: 
    '''Generates decoy protein sequences by reversing every original sequence. Returns a DataFrame with both 
    the original sequences and the original sequences.'''

    decoy_data = copy.deepcopy(data)
    # Reverse each sequence in the database. 
    decoy_data['seq'] = data.seq.apply(lambda s : s[::-1])
    # Label all decoys with an asterisk. 
    decoy_data['gene_id'] = data.gene_id.apply(lambda i : i + '*')

    return pd.concat([data, decoy_data])


def load_genome(strain:str):
    '''Load in the complete nucleotide sequence of the genome.'''
    path = os.path.join(get_ecoli_dir_path(strain), 'genome.fasta')
    with open(path, 'r') as f:
        # lines = f.readlines()[1:] # Skip the header line. 
        lines = f.read().splitlines()[1:] # Skip the header line. 
        seq = ''.join(lines)
    return seq


def load_database(strain:str, remove_decoys:bool=True) -> pd.DataFrame:
    '''Load the database.fasta file in as a pandas DataFrame.'''

    path = os.path.join(get_ecoli_dir_path(strain), 'database.fasta')
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


def load_xtandem(strain:str) -> pd.DataFrame:
    '''Read the result of writing the X! Tandem output to a CSV file.'''
    path = os.path.join(get_ecoli_dir_path(strain), 'xtandem.csv')
    return pd.read_csv(path, index_col=0)


def load_proteins(strain:str) -> pd.DataFrame:
    '''Load the proteins.fasta file in as a pandas DataFrame. This file contains the sequences of the 
    genes predicted by PGAP.'''

    path = os.path.join(get_ecoli_dir_path(strain), 'proteins.fasta')

    with open(path, 'r') as f:
        text = f.read()
        seqs = re.split(r'^>.*', text, flags=re.MULTILINE)[1:]
        # Strip all of the newline characters from the amino acid sequences. 
        seqs = [s.replace('\n', '') for s in seqs]
        # Extract the protein accessions from the header lines. 
        headers = re.findall(r'^>.*', text, re.MULTILINE)
        accessions = [h.split(' ')[0].replace('>', '') for h in headers]

    data = {'ncbi_seq':seqs, 'accession':accessions}
    return pd.DataFrame(data)


def load_predictions(strain:str) -> List[str]:
    '''Load in the gene IDs of the proteins predicted to be selenoproteins, excluding the known
    selenoproteins.'''
    path = os.path.join(get_ecoli_dir_path(strain), 'predictions.csv')
    predictions = pd.read_csv(path).values.ravel()
    # Exclude the known selenoproteins from the list. 
    # known_selenoproteins = ['fdoG', 'fdnG', 'fdhF']
    # return [p for p in predictions if p not in known_selenoproteins]
    return list(predictions)


def load_coordinates(strain:str) -> pd.DataFrame:
    '''Load in the gene coordinates and other metadata as a pandas DataFrame, processing the start and
    stop locations as integers in two separate columns. This assumes the coordinate data was obtained from 
    the www.ncbi.nlm.nih.gov/datasets/gene/. 
    
    :return: A DataFrame with columns for start and stop locations, the RefSeq gene ID, and whether or not
        the gene is read in reverse. Also includes the length of the protein in amino acids for later
        validation. 
    '''
    path = os.path.join(get_ecoli_dir_path(strain), 'coordinates.tsv')

    data = pd.read_csv(path, sep='\t', usecols=['Orientation', 'Begin', 'End', 'Symbol', 'Protein length', 'Protein accession'])
    data = data.rename(columns={'Orientation':'orientation', 'Begin':'nt_start', 'End':'nt_stop', 'Symbol':'gene_id', 'Protein length':'aa_length', 'Protein accession':'accession'}) # Rename the columns for my sanity. 
    data[['nt_start', 'nt_stop']] = data[['nt_start', 'nt_stop']].apply(pd.to_numeric) # Convert the start and stop to integers. 
    data['nt_start'] = data['nt_start'] - 1 # Shift all starts to be zero-indexed for my sanity...
    data['reverse'] = [o == 'minus' for o in data.orientation]
    data = data.drop(columns=['orientation'])
    return data.fillna('none')


def xtandem_run(strain:str) -> NoReturn:
    '''Run X! Tandem on a directory of mgf files. Writes the files to a directory in the ECOLI_DIR_PATH folder.'''
    path = os.path.join(get_ecoli_dir_path(strain), 'ms/mgf/')

    for sample in os.listdir(path):
        sample = sample.replace('.mgf', '') # Remove the file extension. 
        # Read in the input.xml template file, and fill in the sample name. 
        with open('/home/prichter/Documents/selenobot-detect/scripts/input.xml', 'r') as f:
            input_xml = f.read().format(sample=sample, strain=strain)
        with open('tmp.xml', 'w') as f:
            f.write(input_xml)
        
        cmd = '/home/prichter/tandem-linux-17-02-01-4/bin/tandem.exe'
        subprocess.run([cmd, 'tmp.xml'])
        # Remove the temporary input.xml file. 
        subprocess.run(['rm', 'tmp.xml'])


# def xtandem_check(strain:str) -> NoReturn:

#     xtandem = load_xtandem(strain)
#     database = load_database(strain)
#     n, m = len(xtandem), len(database) # Original length of the data. 

#     database = database.drop_duplicates('accession')
#     if len(database) != m: print(f'ecoli.check_xtandem: Dropped {m - len(database)} duplicate accessions.')

#     xtandem = xtandem.merge(database[['nt_start', 'nt_stop', 'accession']], on=['accession'], how='left')
#     assert len(xtandem) == n, f'ecoli.check_xtandem: Merge was unsuccessful. Length before merge {n}, length after merge {len(database)}.'

#     for row in xtandem.itertuples():
#         assert row.nt_domain_start >= row.nt_start and row.nt_domain_start < row.nt_stop
#         assert row.nt_domain_stop > row.nt_start and row.nt_domain_stop <= row.nt_stop
#         overlap_start, overlap_stop = get_overlap(row.nt_start, row.nt_stop, row.nt_domain_start, row.nt_domain_stop)
#         assert (overlap_stop - overlap_start) == (row.nt_domain_stop - row.nt_domain_start)


def xtandem_parse_xml_output(strain:str) -> NoReturn:
    '''Read everything in the X! Tandem output directory into a pandas DataFrame. Writes the
    rerulting output to a CSV file in the strain directory. '''

    def get_coordinates(data:pd.DataFrame) -> pd.DataFrame:
        '''Get the absolute nucleotide coordinates of domain starts and stops. Domain stops and starts are
        originally given in terms of # amino acids from the start of the protein,'''
        database = load_database(strain)
        n, m = len(data), len(database) # Original length of the data. 

        database = database.drop_duplicates('accession')
        if len(database) != m: print(f'ecoli.xtandem_parse_xml_output: Dropped {m - len(database)} duplicate accessions.')

        data = data.merge(database[['nt_start', 'nt_stop', 'accession']], on=['accession'], how='left')
        assert len(data) == n, f'ecoli.xtandem_parse_xml_output: Merge was unsuccessful. Length before merge {n}, length after merge {len(database)}.'

        data['nt_domain_start'] = data.nt_start + (3 * data.aa_domain_start)
        data['nt_domain_stop'] = data.nt_start + (3 * data.aa_domain_stop)
        return data.drop(columns=['nt_start', 'nt_stop'])

    def parse(path:str) -> pd.DataFrame:
        '''Parses the output of an X! Tandem run on a single sample.'''

        # Columns in the DataFrame output. 
        cols = ['gene_id', 'aa_domain_start', 'aa_domain_stop', 'domain_log10_e', 'protein_log10_e', 'domain_seq', 'aa_ext', 'aa_length', 'accession', 'reverse']
        sample_data = {col:[] for col in cols} 

        # parser = etree.XMLParser(recover=True) # Ignore some (inconsequential?) parsing errors. 
        tree = xml.etree.ElementTree.parse(path)
        root = tree.getroot()

        for protein in root.iter('protein'):
            info = {k:v for k, v in [i.split('=') for i in protein.attrib.get('label').split('|')]}
            protein_log10_e = protein.attrib.get('expect')

            # E value describes the number of hits one can “expect” to see by chance when searching a database. 
            for domain in protein.iter('domain'):
                sample_data['protein_log10_e'].append(protein_log10_e)
                sample_data['domain_log10_e'].append(float(domain.attrib.get('expect')))
                sample_data['domain_seq'].append(domain.attrib.get('seq'))
                sample_data['aa_domain_start'].append(int(domain.attrib.get('start')))
                sample_data['aa_domain_stop'].append(int(domain.attrib.get('end')))
                sample_data['gene_id'].append(info['gene_id'])
                sample_data['accession'].append(info['accession'])
                sample_data['reverse'].append(bool(info['reverse']))
                sample_data['aa_ext'].append(int(info['nt_ext']) // 3)
                sample_data['aa_length'].append(int(info['aa_length']))

        sample_data = pd.DataFrame(sample_data)
        sample_data['sample'] = os.path.basename(path).replace('.xml', '') # Add the sample name to the DataFrame.
        return sample_data

    dir_path = os.path.join(get_ecoli_dir_path(strain), 'xtandem')
    data = []
    pbar = tqdm([f for f in os.listdir(dir_path) if '.xml' in f], desc='ecoli.xtandem_parse_xml_output')
    for filename in pbar:
            pbar.set_description(f'ecoli.xtandem_parse_xml_output: Reading output from {filename}.')
            data.append(parse(os.path.join(dir_path, filename) ))

    data = get_coordinates(pd.concat(data)) # Add the absolute nucleotide positions of the domain hits. 
    data.to_csv(os.path.join(get_ecoli_dir_path(strain), 'xtandem.csv'))

        
def xtandem_get_fpr(strain:str, threshold:float=1) -> float:
    '''Calculate the false positive rate of an X! Tandem run using the decoy sequences.'''
    xtandem = load_xtandem(strain)
    # NOTE: I think this might need to be the expectation value for the whole protein, but not totally sure?
    xtandem = xtandem[xtandem['protein_log10_e'] <= threshold] # Filter according to E value threshold.
    # Decoy sequences are flagged with an asterisk. 
    return sum(xtandem.gene_id.str.contains('*', regex=False)) / len(xtandem)


def xtandem_get_hits_past_original_stop(strain:str) -> pd.DataFrame:
    '''Looks at the original database and location of X! Tandem hits to determine if the hit
    lies outside the original range of the protein.'''

    predictions = load_predictions(strain)
    output = load_xtandem(strain)
    # Extract all hits for the predicted selenoproteins. 
    output = output[output.gene_id.isin(predictions)]
    return output[output.apply(lambda row : (row.aa_length - row.aa_ext) < row.aa_domain_stop, axis=1)]


def database_size(strain:str, decoy:bool=False) -> int:
    '''Get the number of genes in the FASTA database.'''
    path = os.path.join(get_ecoli_dir_path(strain), 'database.fasta')

    ids = fasta_ids(path, get_id_func=lambda i : i.replace('|', ''))
    if not decoy:
        return len([i for i in ids if '*' not in i])
    else:
        return len(ids)


def get_overlap(start_a:int, stop_a:int, start_b:int, stop_b:int) -> Tuple[int, int]:
    '''Calculate any overlap between two sequences in units of nucleotides. Assumes the sequences flanked by the 
    start and stop locations are located on the same strand.'''
    if (start_a <= start_b) and (stop_a > start_b) and (stop_a <= stop_b):
        # (start_a) ... (start_b) ... (stop_a) ... (stop_b)
        return start_b, stop_a
    if (start_b <= start_a) and (stop_b > start_a) and (stop_b <= stop_a):
        # (start_b) ... (start_a) ... (stop_b) ... (stop_a)
        return start_a, stop_b
    elif (start_a <= start_b) and (stop_a >= stop_b): # If the entire sequence is overlapping. 
        # (start_a) ... (start_b) ... (stop_b) ... (stop_a)
        return start_b, stop_b
    elif (start_b <= start_a) and (stop_b >= stop_a): # If the entire sequence is overlapping. 
        # (start_b) ... (start_a) ... (stop_a) ... (stop_b)
        return start_a, stop_a
    else:
        return None, None


def database_check(strain:str):
    '''Run a series of checks on the database.'''

    # The following code does not account for decoys.  
    predictions = load_predictions(strain)
    database = load_database(strain, remove_decoys=True)
    proteins = load_proteins(strain)

    n = len(database) # Original size of database. 

    repeated_accessions = list(database.accession.value_counts()[database.accession.value_counts() > 1].index)
    assert len(repeated_accessions) == 0, f'database.database_check: Values in the accession column are not unique. {len(repeated_accessions)} instances of repeats.'

    def get_residue_difference(seq:str, ncbi_seq:str) -> int:
        '''Calculate the difference between the translated and NCBI reference sequence, assuming equal length.'''
        return sum([r1 != r2 for r1, r2 in zip(seq, ncbi_seq)])

    def get_overlaps(nt_start:int, nt_stop, database:pd.DataFrame) -> str:
        '''Determines whether or not there is overlap between the sequence in the given range and other
        proteins in the databasebase.'''
        overlaps = []
        for row in database.itertuples():
            overlap_start, overlap_stop = get_overlap(nt_start, nt_stop, row.nt_start, row.nt_stop)
            if overlap_start is not None:
                overlaps.append(f'{row.gene_id} ({overlap_stop - overlap_start})') # Record the amount of overlap. 
        return ', '.join(overlaps)

    database = database.merge(proteins, on=['accession'])
    database = database.drop_duplicates(['nt_stop', 'nt_start']) # Duplicates, for some reason? I think this means multiple accessions?
    assert len(database) == n, f'ecoli.database_check: Merge was unsuccessful. Length before merge {n}, length after merge {len(database)}.'

    translation_errors = {}
    length_mismatches = {}
    extended_sequence_overlaps = {}
    
    for row in database.itertuples():
        # Special case for manually-extended sequences. 
        if (row.gene_id in predictions) and (row.gene_id not in known_selenoproteins):
            # For the extended sequences, be sure to check for overlap. Exclude the gene itself, and those on the opposite strand. 
            overlaps = get_overlaps(row.nt_start, row.nt_stop, database[(database.accession != row.accession) & (database.reverse == row.reverse)])
            if len(overlaps) > 0:
                extended_sequence_overlaps[row.gene_id] = overlaps
            continue # Skip the remaining checks, which do not work if the genes have been extended past the forst stop codon. 
        
        if len(row.ncbi_seq) != len(row.seq):
            length_mismatches[row.gene_id] = len(row.ncbi_seq) - len(row.seq)
        elif row.ncbi_seq != row.seq:
            diff = get_residue_difference(row.seq, row.ncbi_seq)
            if diff > 1: # Noticed that start codons are sometimes mistranslated, but shouldn't affect much, so ignore. 
                translation_errors[row.gene_id] = diff

    if len(translation_errors) > 0: print('Translation errors > 1 AA:', '\n'.join([f'\t{k} {v}' for k, v in translation_errors_greater_than_one.items()]))
    if len(length_mismatches) > 0: print('Length mismatches:\n', '\n'.join([f'\t{k} {v}' for k, v in length_mismatches.items()]))
    if len(extended_sequence_overlaps) > 0: print('Overlaps in extended sequences:\n', '\n'.join([f'\t{k}: {v}' for k, v in extended_sequence_overlaps.items()]))



def database_write(data:pd.DataFrame, strain:str) -> NoReturn:
    '''Write the search database to a FASTA file.'''
    text = ''
    with open(os.path.join(get_ecoli_dir_path(strain), 'database.fasta'), 'w') as f:
        for row in data.itertuples():
            text += f'>gene_id={row.gene_id}|nt_ext={row.nt_ext}|aa_length={row.aa_length}|nt_start={row.nt_start}|nt_stop={row.nt_stop}|accession={row.accession}|reverse={row.reverse}\n'

            # Split the sequence up into shorter, 60-character strings.
            n = len(row.seq)
            seq = [row.seq[i:min(n, i + 80)] for i in range(0, n, 80)]
            assert len(''.join(seq)) == n, 'ecoli.database_write: Part of the sequence was lost when splitting into lines.'
            text += '\n'.join(seq) + '\n'
        
        f.write(text)


def database_build(strain:str) -> NoReturn:
    '''Build the X! Tandem search database using the information from the coordinates.tsv file and the original genome.'''
    # Grab the coordinate information about the predicted selenoproteins only. 
    database = load_coordinates(strain)
    predictions = load_predictions(strain)
    # Mark the sequences which will be extended past the first STOP codon. 
    database['extend'] = [(gene_id in predictions) and (gene_id not in known_selenoproteins) for gene_id in database.gene_id]
    database = get_sequences(database, load_genome(strain))
    database = get_decoy_sequences(database)
    database_write(database, strain)



class Protein():
    
    def __init__(self, gene_id:str, strain:str='mg1655'):
        '''Inititalize a Protein object.'''
        
        database = load_database(strain)
        assert gene_id in list(database.gene_id.values), f'ecoli.Protein: {gene_id} is not present in the database.' 
        row = database[database.gene_id == gene_id] # The row for the specific protein. 
    
        if len(row) > 1: #TODO: Fix this!
            row = row.head(1)

        xtandem = load_xtandem(strain)

        self.nt_start = row.nt_start.item() 
        self.nt_stop = row.nt_stop.item()
        self.gene_id = gene_id
        self.accession = row.accession.item()
        self.reverse = row.reverse.item()
        self.seq = row.seq.item()
        self.length = len(self.seq)
        self.hits = xtandem[xtandem.gene_id == gene_id] # Filter the X! Tandem hits for the particular protein. 
        self.overlaps = self.get_overlaps(xtandem)
        self.strain = strain

    def get_overlaps(self, xtandem:pd.DataFrame) -> pd.DataFrame:
        '''Find the locations where the sequence overlaps with other sequence in the database.
        Returns a subset of the X! Tandem results DataFrame with hits which overlap the Protein.'''
        xtandem = xtandem[xtandem.accession != self.accession] # Exclude hits belonging to the Protein.
        xtandem = xtandem[xtandem.reverse == self.reverse] # Exclude hits belonging to genes on the opposite strand.

        f = lambda r : get_overlap(self.nt_start, self.nt_stop, r.nt_domain_start, r.nt_domain_stop)
        xtandem[['nt_overlap_start', 'nt_overlap_stop']] = xtandem[['nt_domain_start', 'nt_domain_stop']].apply(f, axis=1, result_type='expand')

        xtandem = xtandem.dropna(axis=0) # Drop rows where no overlap was detected. 
        # Convert overlap to units of amino acids relative to start of the protein. 
        xtandem['overlap_start'] = (xtandem.nt_overlap_start - self.nt_start) // 3
        xtandem['overlap_stop'] = (xtandem.nt_overlap_stop - self.nt_start) // 3
        # Restrict overlaps to the range of the protein. 
        xtandem['overlap_start'] = xtandem.overlap_start.apply(lambda n : max(0, n))
        xtandem['overlap_stop'] = xtandem.overlap_stop.apply(lambda n : min(n, self.length))
        return xtandem.drop(columns=['nt_overlap_start', 'nt_overlap_stop'])


    def __len__(self) -> int:
        '''Return the length of the protein sequence.'''
        return self.length

    def __str__(self) -> str:
        '''Set the string representation of the Protein object to be the amino acid sequence.'''
        return self.seq

    def plot(self, plot_overlaps:bool=True, show:bool=False, save:bool=True) -> NoReturn:
        '''Visualize the X! Tandem hits on the protein sequence.'''

        if len(self.hits) < 1:
            print(f'ecoli.Protein.plot: No hits found for protein {self.gene_id}.')
            return

        fig, ax = plt.subplots(1)
        legend = [self.gene_id]

        points = {}
        for hit in self.hits.itertuples():
            for x in range(hit.aa_domain_start, hit.aa_domain_stop + 1):
                if x in points:
                    points[x] += 1
                else:
                    points[x] = 1
        x_vals = list(range(self.length))
        y_vals = [points.get(x, 0) for x in x_vals]
        ax.plot(x_vals, y_vals)

        if plot_overlaps and (len(self.overlaps) > 0):
            transform = ax.get_xaxis_transform() # Get the coordinate system of the x-axis. 

            for gene_id in self.overlaps.gene_id.drop_duplicates(): # Overlaps in units amino acid. 
                legend.append(gene_id)
                overlap_hits = self.overlaps[self.overlaps.gene_id == gene_id]
                points = {}
                for hit in overlap_hits.itertuples():
                    for x in range(hit.aa_domain_start, hit.aa_domain_stop + 1):
                        if x in points:
                            points[x] += 1
                        else:
                            points[x] = 1
                x_vals = list(range(self.length))
                y_vals = [points.get(x, 0) for x in x_vals]
                ax.plot(x_vals, y_vals)

        ax.set_xticks(np.arange(self.length + 1))
        ax.set_xticklabels(['' if aa != 'U' else aa for aa in self.seq] + ['STOP'])
        ax.set_yticks(np.arange(0, max(y_vals)))
        ax.set_ylabel('number of hits')
        ax.set_title(self.gene_id)

        ax.legend(legend)

        # fig.subplots_adjust(bottom=0.5)
        fig.tight_layout()

        if show:
            plt.show()
        if save:
            path = os.path.join(get_ecoli_dir_path(self.strain), 'plots', f'{self.gene_id}.png')
            fig.savefig(path, format='png')


# TODO: Add color-coding by E-value, perhaps?
# TODO: Add potential overlap with other sequences. 
# TODO: Possibly change how the hits are plotted, maybe to look more like a spectrum. 

# The goal is to find hits which are in the region after the stop codon of the predicted proteins, but don't overlap with the next protein. 
if __name__ == '__main__':

    # database_build('mg1655')
    # database_build('bw25113')

    # xtandem_run('mg1655')
    # xtandem_run('bw25113')
    # xtandem_parse_xml_output('mg1655')
    # xtandem_parse_xml_output('bw25113')


    for gene_id in load_predictions('bw25113'):
        try:
            protein = Protein(gene_id, strain='bw25113')
            protein.plot(save=True)
        except:
            continue

# eco:b3894 seems to have an extra in-frame stop codon which I don't quite understand. It also has a U present, which means
# that this is not the issue.
# The issue seems to be that there is already a stop codon in the gene boundaries set in the file (i.e. already known selenoproteins).
# Will need to add a special case for this.  