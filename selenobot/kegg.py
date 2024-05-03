'''Code for downloading sequences from KEGG and parsing files.'''
import requests
import re
import tqdm
from typing import List, Dict
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from typing import NoReturn

def kegg_parse_genes(genes:List[str]):
    parsed_genes = []
    for gene in genes:
        organism, entries = gene.split(':')
        entries = re.sub('\([^)]*\)', '', entries) # Remove anything in parentheses (stuff like (fdnG)).
        entries = entries.strip().split()
        parsed_genes += [f'{organism.lower()}:{entry}' for entry in entries]
    return parsed_genes


def kegg_parse_position(position:str):

    start, stop, orientation = None, None, None
    if re.search('^complement\((\d+)\.\.(\d+)\)', position) is not None:
        match = re.search('complement\((\d+)\.\.(\d+)\)', position)
        start = int(match.group(1))
        stop = int(match.group(2))
        orientation = '-'
    elif re.search('(\d+)\.\.(\d+)', position) is not None:
        match = re.search('(\d+)\.\.(\d+)', position)
        orientation = '+'
        start = int(match.group(1))
        stop = int(match.group(2))
    else: 
        print(f'kegg_parse_position: Could not extract position from string {position}.')
    return start, stop, orientation


def kegg_parse_file_ko(text:str=None, path:str=None) -> Dict:

    if path is not None: # IF a path is specified, read in text from a file.
        with open(path, 'r') as f:
            text = f.read()
    
    info = {'symbol':None, 'genes':None}
    text = text.split('\n') # Split the returned text into lines. 
    i = 0
    while i < len(text):
        if 'SYMBOL' in text[i]:
            symbol = text[i].split()[-1]
            info['symbol'] = symbol
            i += 1
        elif 'GENES' in text[i]:
            # First gene entry has both GENE and the gene name. 
            genes = []
            genes.append(text[i].replace('GENES', '').strip())
            i += 1
            while re.match('^ {6}.+', text[i]) is not None:
                genes.append(text[i].strip())
                i += 1
            genes = kegg_parse_genes(genes)
            info['genes'] = genes
        else:
            i += 1
    return info


def kegg_parse_file_gene(text:str=None, path:str=None) -> Dict:

    if path is not None: # IF a path is specified, read in text from a file.
        with open(path, 'r') as f:
            text = f.read()
    info = {'start':None, 'stop':None, 'orientation':None, 'aa_seq':None, 'nt_seq':None}
    text = text.split('\n')
    i = 0
    while i < len(text):
        if 'POSITION' in text[i]:
            position = text[i].replace('POSITION', '').strip()
            start, stop, orientation = kegg_parse_position(position.strip())
            info['start'], info['stop'], info['orientation'] = start, stop, orientation
            i += 1
        elif 'ORGANISM' in text[i]:
            organism = text[i].replace('ORGANISM', '').strip()
            info['organism'] = organism
            i += 1
        elif 'NTSEQ' in text[i]:
            length = int(text[i].replace('NTSEQ', '').strip())
            i += 1 # Skip the line with the length. 
            nt_seq = ''
            while re.match('^ {6}.+', text[i]) is not None:
                nt_seq += text[i].strip()
                i += 1
            assert len(nt_seq) == length, f'kegg_parse_file_gene: The length of the nucleotide sequence is incorrect. Expected {length}, but sequence is of length {len(nt_seq)}.'
            info['nt_seq'] = nt_seq
        elif 'AASEQ' in text[i]:
            length = int(text[i].replace('AASEQ', '').strip())
            i += 1 # Skip the line with the length. 
            aa_seq = ''
            while re.match('^ {6}.+', text[i]) is not None:
                aa_seq += text[i].strip()
                i += 1
            assert len(aa_seq) == length, f'kegg_parse_file_gene: The length of the amino acid sequence is incorrect. Expected {length}, but sequence is of length {len(aa_seq)}.'
            info['aa_seq'] = aa_seq
        else:
            i += 1
    return info


def kegg_get_genome(organism:str, start:int=1, stop:int=None, complement:bool=False) -> str:
    '''Get a genome, or a portion of the genome, for the specified organism.'''
    if stop is None:
        # If no stop is specified, I think it just goes to the end of the genome sequence. 
        url = f'https://www.genome.jp/dbget-bin/cut_sequence_genes.pl?FROM={start}&ORG={organism}&VECTOR={1 if not complement else -1}'
    else:
        url = f'https://www.genome.jp/dbget-bin/cut_sequence_genes.pl?FROM={start}&TO{stop}&ORG={organism}&VECTOR={1 if not complement else -1}'

    def is_nucleotides(s:str) -> bool:
        return re.match('^[ATGC]+$', s) is not None
    
    def is_header(s:str) -> bool:
        return '>' in s

    # Get the genome and parse the returned HTML text string (i.e. remove font color and titles).
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    parsed = soup.get_text().split('\n')
    parsed = [line for line in parsed if (is_nucleotides(line) or is_header(line))]
    
    return '\n'.join(parsed)


def kegg_get_genomes_by_ko(ko:str, genes_path:str=None, output_path:str='/home/prichter/Documents/selenobot/data/validation/v2/genomes'):
    assert output_path is not None, 'kegg_get_genomes_by_ko: A path for the genomes must be specified.'
    if genes_path is not None:
        genes_df = pd.read_csv(genes_path)
    else:
        genes_df = kegg_get_genes_by_ko(ko)
    # Get the organism codes from the DataFrame. 
    organisms = set(genes_df.gene_id.str.split(':', expand=True)[0])
    for organism in organisms:
        if f'{organism}.fn' in os.listdir(output_path):
            print(f'kegg_get_genomes_by_ko: Genome of organism {organism} is already downloaded.')
            continue
        print(f'kegg_get_genomes_by_ko: Downloading genome for organism {organism}.')
        genome = kegg_get_genome(organism)
        with open(os.path.join(output_path, f'{organism}.fn'), 'w') as f:
            f.write(genome)


kegg_get_genomes_by_ko('K08348', genes_path='~/Documents/selenobot/data/validation/v2/K08348.csv')


def kegg_get_genes_by_ko(ko:str, n:int=None):
    url = f'https://rest.kegg.jp/get/ko:{ko}'
    text = requests.get(url).text
    assert text is not None, 'kegg_get_genes_by_ko: Something went wrong with HTTP request to KEGG.'
    ko_info = kegg_parse_file_ko(text=text)

    df = []
    genes = ko_info['genes'] if (n is None) else ko_info['genes'][:n]
    for gene in tqdm.tqdm(genes, desc='kegg_get_genes_by_ko: Retrieving genes from KEGG...'):
        url = f'https://rest.kegg.jp/get/{gene}'
        text = requests.get(url).text
        assert text is not None, 'kegg_get_genes_by_ko: Something went wrong with HTTP request to KEGG.'
        row = kegg_parse_file_gene(text=text)
        row['gene_id'] = gene
        df.append(row)
    df = pd.DataFrame(df)
    df['ko'] = ko
    return df


def kegg_check_genome(path:str) -> NoReturn:
    pass
