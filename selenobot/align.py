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
from io import StringIO


DATA_DIR = '/home/prichter/Documents/data/selenobot/validation/homology/'
blastp = '/home/prichter/ncbi-blast-2.15.0+/bin/blastp' # Location of blast binary. 
makeblastdb = '/home/prichter/ncbi-blast-2.15.0+/bin/makeblastdb' 


def mmseqs_load_sequence_from_database(gene_id:str) -> str:
    '''Grab a specific sequence from the query database stored in the query.fasta file.'''
    database = mmseqs_load_database(path=os.path.join(DATA_DIR, 'query.fasta')) # Not really working with control sequences right now.
    database = database[database.gene_id == gene_id]
    # This should only have one result -- the gene_id should be unique in the query database.
    assert len(database) < 2, 'mmseqs.load_database_sequence: The gene_ids in the database should be unique.'
    assert len(database) > 0, 'mmseqs.load_database_sequence: The gene_id is not present in the query.fasta query database.'
    return database.seq.item() 


def blast_run(query_filename:str, target_filename:str, output_filename:str) -> NoReturn:
    '''Run blastp on the specified query and target files, and write the result to the output filename. 
    The output will be in customized CSV format.''' 

    query_path = os.path.join(DATA_DIR, 'alignments', query_filename)
    target_path = os.path.join(DATA_DIR, 'alignments', target_filename)
    output_path = os.path.join(DATA_DIR, 'alignments', output_filename)

    # Use a custom format, which is equivalent to fmt 6, but with the addition of the aligned parts of the query and subject sequence. 
    fmt = '"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq"'
    subprocess.run(f'{blastp} -query {query_path} -subject {target_path} -out {output_path} -outfmt {fmt}', shell=True, check=True)


def blast_build_query(query_gene_id:str, query_seq:str, filename:str) -> NoReturn:
    '''Create a query file for a BLAST alignment using the input sequence data The query file should be
    in FASTA format, and will contain a single sequence (in my case).'''
    fasta = f'>query_gene_id={query_gene_id}'
    fasta += '\n' + query_seq

    path = os.path.join(DATA_DIR, 'blast', filename)
    with open(path, 'w') as f:
        # Write the FASTA string to a file. 
        f.write(fasta)


def blast_build_target(data:pd.DataFrame, filename:str) -> NoReturn:
    '''Create a target database for a BLAST alignment using the sequences in the input DataFrame.'''
    fasta = ''
    for row in data.itertuples():
        header_fields = ['target_gene_id', 'query_domain_start', 'query_domain_stop', 'target_domain_start', 'target_domain_stop']
        header = '>' + '|'.join([field + '=' + str(getattr(row, field)) for field in header_fields])
        fasta += header + '\n'
        fasta += row.target_seq + '\n'

    path = os.path.join(DATA_DIR, 'blast', filename)
    with open(path, 'w') as f:
        # Write the FASTA string to a file. 
        f.write(fasta)

    # Convert the FASTA file to a blast database. What does the parse_seqids option do?
    # subprocess.run(f'{makeblastdb} -in {path} -dbtype prot -parse_seqids', shell=True, check=True)


def mmseqs_create_results_csv(path:str=os.path.join(DATA_DIR, 'mmseqs', 'mmseqs.csv')) -> NoReturn:
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

    data.to_csv(path)


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



