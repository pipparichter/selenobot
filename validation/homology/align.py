'''Code for searching for analyzing results of MMSeqs2 homology search.'''
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
import requests
from Bio import Align
import time

DATA_DIR = '/home/prichter/Documents/data/selenobot/validation/homology/'


def load_database(filename='query.fasta') -> pd.DataFrame:
    '''Load the database FASTA file in as a pandas DataFrame.'''

    path = os.path.join(DATA_DIR, filename)
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

    return data


def load_sequence_from_database(gene_id:str) -> str:
    '''Grab a specific sequence from the query database stored in the query.fasta file.'''
    database = load_database(filename='query.fasta') # Not really working with control sequences right now.
    database = database[database.gene_id == gene_id]
    # This should only have one result -- the gene_id should be unique in the query database.
    assert len(database) < 2, 'mmseqs.load_database_sequence: The gene_ids in the database should be unique.'
    assert len(database) > 0, 'mmseqs.load_database_sequence: The gene_id is not present in the query.fasta query database.'
    return database.seq.item() 



def mmseqs_create_csv() -> NoReturn:
    '''Organize the MMSeqs hit results from the search of GTDB into a pandas DataFrame, and add some
    extra information. Also download amino acid sequences using Find-A-Bug. Write the resulting DataFrame 
    to a CSV file.'''
    # Ignore the controls for the time being. 
    dir_path = os.path.join(DATA_DIR, 'mmseqs', 'results_s_default')

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

    # NOTE: Requiring the hit to SPAN the stop codon does not seem to make a difference. 
    def label_hits_past_stop_codon(data:pd.DataFrame, require_span:bool=True) -> pd.DataFrame: #, remove_query_gene_ids:List[str]=[]) -> pd.DataFrame:
        '''Find instances where a domain match is found past the first stop codon. The aa_length column gives the length of the
        extended amino acid sequence (if an extension was applied), so the stop codon location can be computed by subtracting the aa_ext
        from the length.
        
        :param data: The DataFrame containing the BLAST hits for the query sequences against GTDB.
        :param require_span: Whether or not to only include hits which span the U codon (not just in the region to the right of it).
        :return: A DataFrame which contains the hits which meet the parameter specifications. 
        '''
        if require_span: 
            # Make sure the beginning of the matching domain is to the left of the putative U position, AND the end of the match is to the right.
            data['past_stop_codon'] = (data.query_domain_start <= data.query_u_position) & (data.query_domain_stop >= data.query_u_position)
        else: # Filter by locations where the end of the matching domain in the query sequence is found past the extension.
            data['past_stop_codon'] = data.query_domain_stop >= data.query_u_position
        return data 
    
    # NOTE: Must be connected to the VPN for this to work. 
    def download_sequences(data:pd.DataFrame) -> pd.DataFrame:
        '''Download amino acid sequences for each target in the DataFrame.'''
        seqs = []
        for target_gene_id in tqdm(data.target_gene_id, desc='main.mmseqs_create_csv'): # These are the GTDB accessions. 
            url = f'http://microbes.gps.caltech.edu:8000/api/sequences?gene_id={target_gene_id}'
            # print(f'main.mmseqs_create_csv: Downloaded sequence for gene {target_gene_id}.')
            # time.sleep(2) # Find-A-Bug gets overwhelmed, so need to pause between queries. 
            response = requests.get(url).text.strip()
            seq = response.split('\n')[-1].split(',')[-1]
            seqs.append(seq)
        data['target_seq'] = seqs
        return data

    # Group the data from all the target database splits.
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

    # Compute the location of the posited U residue. 
    data['query_u_position'] = data.query_aa_length - data.query_aa_ext
    data = label_hits_past_stop_codon(data)

    # Remove the things we know to be false positives. 
    data = data[~data.query_gene_id.isin(['ilvB', 'ivbL'])]
    # Download the amino acid sequences and add them to the DataFrame. 
    data = download_sequences(data)

    data.to_csv(os.path.join(DATA_DIR, 'mmseqs.csv'))


def load_mmseqs() -> pd.DataFrame:
    '''Read in the DataFrame generated by mmseqs_create_csv.'''
    path = os.path.join(DATA_DIR, 'mmseqs.csv')
    return pd.read_csv(path, index_col=0)


def load_alignments(filename:str) -> pd.DataFrame:
    '''Read in the DataFrame generated by get_pairwise_alignments.'''
    path = os.path.join(DATA_DIR, 'alignments', filename)
    return pd.read_csv(path, index_col=0)


def plot_hits(data:pd.DataFrame, query_gene_id:str, past_stop_codon:bool=False) -> NoReturn:
    '''Visualize the homology hits on the query sequence.'''

    if past_stop_codon:
        data = data[data.past_stop_codon]

    fig, ax = plt.subplots(1)
    assert query_gene_id in data.query_gene_id.values, f'homology.plot_hits_past_stop_codon: {query_gene_id} is not present in the data.'

    data = data[data.query_gene_id.str.match(query_gene_id)]
    u_loc = data.query_u_position.iloc[0] # Get the location of the putative selenocysteine residue. 

    locs = np.arange(data.query_aa_length.iloc[0])
    locs_labels = ['' if (l != u_loc) else 'U' for l in locs]

    hits = {i:0 for i in locs} # Create a map from amino acid position to number of hits. 
    for hit in d.itertuples():
        for loc in range(hit.query_domain_start, hit.query_domain_stop + 1):
            hits[loc] += 1
    ax.plot(locs, [hits[x] for x in locs])

    # ax.legend(['result', 'control'])
    ax.set_title(query_gene_id if not past_stop_codon else f'{query_gene_id} (hits past stop codon)')
    ax.set_xticks(locs)
    ax.set_xticklabels(locs_labels)
    ax.set_xlabel('residue position')
    ax.set_ylabel('hits')

    plt.savefig(f'{query_gene_id}.png' if not past_stop_codon else f'{query_gene_id}_past_stop_codon.png', format='PNG')



# NOTE: Calvin said pairwise alignment is sufficient -- no need for MSA. 
def get_pairwise_alignments(data:pd.DataFrame, filename:str='mmseqs_alignments.csv', max_num_alignments:int=1000) -> NoReturn:
    '''Generate multisequence alignments for the query sequence against each target sequence in the input DataFrame.
    Write the resulting alignments to a file.'''
    # NOTE: Note that pairwise aligners can return an astronomical number of alignments if they align poorly to each other. 
    align_data = []
    skipped = [] # Store the sequences for which an alignment could not be computed. 

    for query_gene_id in set(data.query_gene_id):
        query_seq = load_sequence_from_database(query_gene_id)
        for row in data[data.query_gene_id == query_gene_id].itertuples():
            # There are multiple possible alignments, so calling align returns a list. 
            aligner = Align.PairwiseAligner()
            try:
                alignments = aligner.align(query_seq, row.target_seq)
                if max_num_alignments:
                    print(f'homology.get_pairwise_alignments: Reduced number of alignments from {len(alignments)} to {max_num_alignments}.')
                    alignments = [alignments[i] for i in range(min(max_num_alignments, len(alignments)))]
            except OverflowError:
                skipped.append(row.target_gene_id)
                continue

            for a in alignments:
                query_align, target_align = a.format('phylip').strip().split('\n')[1:]
                query_align, target_align = query_align.strip(), target_align.strip() # Remove the extra whitespace.
                # assert len(query_align) == len(query_seq.replace('*', '')), 'main.get_pairwise_alignments: Query sequence and query alignment lengths do not match'
                align_data.append({'score':a.score, 'query_align':query_align, 'target_align':target_align, 'target_gene_id':row.target_gene_id, 'query_id':query_gene_id, 'target_seq_length':len(row.target_seq)})

    print(f'homology.get_pairwise_alignments: Unable to generate alignments for {len(skipped)} target genes.')
    align_data = pd.DataFrame(align_data)
    align_data.to_csv(os.path.join(DATA_DIR, filename))

