import requests
import re
import tqdm
from typing import List, Dict
import pandas as pd
import os
import re
from selenobot.utils import DATA_DIR
from bs4 import BeautifulSoup
from typing import NoReturn
from io import StringIO
import subprocess 
import shutil 
import zipfile


def get_genome_with_ncbi_id(genome_id:str, path:str=None, organism:str=None):
    '''Use the NCBI ID (RefSeq ID) to download a genome using the NCBI Datasets tool.'''

    archive_path = os.path.join(path, 'ncbi_dataset.zip')

    def extract_genome_from_archive(genome_id:str):
        # https://stackoverflow.com/questions/4917284/extract-files-from-zip-without-keeping-the-structure-using-python-zipfile
        archive = zipfile.ZipFile(archive_path)
        for member in archive.namelist():
            if member.startswith(f'ncbi_dataset/data/{genome_id}'):
                source = archive.open(member)
                # NOTE: Why does wb not result in another zipped file being created?
                with open(os.path.join(path, f'{organism}.fn'), 'wb') as target:
                    shutil.copyfileobj(source, target)

    # Make sure to add a .1 back to the genome accession (removed while removing duplicates).
    cmd = f'datasets download genome accession {genome_id} --filename {archive_path} --include genome --no-progressbar'
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    extract_genome_from_archive(genome_id)

    os.remove(archive_path)


def parse_genes(genes:List[str]):
    parsed_genes = []
    for gene in genes:
        gene = re.sub('\((.*?)\)', '', gene) # Remove anything in parentheses (stuff like (fdnG)).
        organism, entries = gene.split(':')
        entries = entries.strip().split()
        # There can be multiple genes per organism, so this accounts for this. 
        parsed_genes += [f'{organism.lower()}:{entry}' for entry in entries]
    return parsed_genes


def parse_position(position:str):

    # TODO: This currently doesn't handle plasmids! In this case, locations are given by plasmid_id:{position}

    # One of the metadata files has an error in it -- this is just to handle that special case. 
    position = position.replace('>', '')

    start, stop, strand = None, None, None
    if re.search('^complement\((\d+)\.\.(\d+)\)', position) is not None:
        match = re.search('complement\((\d+)\.\.(\d+)\)', position)
        start = int(match.group(1))
        stop = int(match.group(2))
        strand = '-'
    elif re.search('(\d+)\.\.(\d+)', position) is not None:
        match = re.search('(\d+)\.\.(\d+)', position)
        strand = '+'
        start = int(match.group(1))
        stop = int(match.group(2))
    else: 
       # print(f'parse_position: Could not extract position from string {position}.')
       pass
    return start, stop, strand


def parse_ko_file(text:str=None, path:str=None) -> Dict:

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
            # Read in all of the genes in the list.
            while re.match('^ {6}.+', text[i]) is not None:
                genes.append(text[i].strip())
                i += 1
            genes = parse_genes(genes)
            info['genes'] = genes
        else:
            i += 1
    return info

def parse_genome_file(text:str) -> Dict:
    info = {'organism':None, 'ncbi_id':None, 'complete':None}
    text = text.split('\n')
    i = 0
    while i < len(text):
        if 'DATA_SOURCE' in text[i]: # Get the code for the genome ID. 
            match = re.search(r'GC[AF]_\d{9}\.\d+', text[i])
            info['ncbi_id'] = match.group(0)
            info['complete'] = 'Complete' in text[i]
            i += 1
        elif 'ORG_CODE' in text[i]:
            organism = text[i].replace('ORG_CODE', '').strip()
            info['organism'] = organism
            i += 1 
        else:
            i += 1
    return info

# TODO: Is there an easier/cleaner way to parse these kind of files?
def parse_gene_file(text:str) -> Dict:

    info = {'start':None, 'stop':None, 'strand':None, 'aa_seq':None, 'nt_seq':None, 'genome_id':None, 'ncbi_id':None, 'uniprot_id':None}
    text = text.split('\n')
    i = 0
    while i < len(text):
        if 'ENTRY' in text[i]: # Get the code for the genome ID. 
            genome_id = 'gn:' + text[i].split()[-1].strip()
            info['genome_id'] = genome_id
            i += 1
        if 'ORTHOLOGY' in text[i]:
            ko = text[i].split()[1]
            ko_description = ' '.join(text[i].split()[2:])
            info['ko'], info['ko_description'] = ko, ko_description
            i += 1
        elif 'POSITION' in text[i]:
            position = text[i].replace('POSITION', '').strip()
            start, stop, strand = parse_position(position.strip())
            info['start'], info['stop'], info['strand'] = start, stop, strand
            i += 1
        elif 'DBLINKS' in text[i]:
            while 'AASEQ' not in text[i]: # Until you reach the next secion of the file, which is AASEQ.
                if 'NCBI-ProteinID' in text[i]:
                    info['ncbi_id'] = text[i].replace('NCBI-ProteinID:', '').replace('DBLINKS', '').strip()
                    i += 1
                elif 'UniProt' in text[i]:
                    info['uniprot_id'] = text[i].replace('UniProt:', '').replace('DBLINKS', '').strip()
                    i += 1
                else:
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
            assert len(nt_seq) == length, f'parse_gene_file: The length of the nucleotide sequence is incorrect. Expected {length}, but sequence is of length {len(nt_seq)}.'
            info['nt_seq'] = nt_seq
        elif 'AASEQ' in text[i]:
            length = int(text[i].replace('AASEQ', '').strip())
            i += 1 # Skip the line with the length. 
            aa_seq = ''
            while re.match('^ {6}.+', text[i]) is not None:
                aa_seq += text[i].strip()
                i += 1
            assert len(aa_seq) == length, f'parse_gene_file: The length of the amino acid sequence is incorrect. Expected {length}, but sequence is of length {len(aa_seq)}.'
            info['aa_seq'] = aa_seq
        else:
            i += 1
    return info


def get_metadata_by_organism(organism:str):

    url = f'https://rest.kegg.jp/list/{organism}'
    text = requests.get(url).text 

    columns = ['id', 'seq_type', 'position', 'description']
    df = pd.read_csv(StringIO(text), delimiter='\t', names=columns, header=None)

    position_df = []
    for position in df.position:
        row = dict()
        row['start'], row['stop'], row['strand'] = parse_position(position)
        position_df.append(row)
    position_df = pd.DataFrame(position_df)
    
    df = df.drop(columns=['position'])
    df = pd.concat([position_df, df], axis=1)
    df = df[df['seq_type'] == 'CDS']
    return df


def get_genome(genome_id:str, path:str=os.path.join(DATA_DIR, 'val2', 'genomes'), complete_only:bool=True) -> str:
    '''Get a genome for the specified organism.'''
    # If no stop is specified, I think it just goes to the end of the genome sequence. 
    url = f'https://rest.kegg.jp/get/{genome_id}'
    text = requests.get(url).text
    info = parse_genome_file(text)

    organism = info['organism']
    if (not complete_only) or info['complete']:
        get_genome_with_ncbi_id(info['ncbi_id'], path=path, organism=organism)
        print(f"get_genome: Saved genome to {os.path.join(path, f'{organism}.fn')}")
    else:
        print(f'get_genome: Skipping genome for organism {organism}, which is not a complete genome.')



# def get_genome(organism:str) -> str:
#     '''Get a genome for the specified organism.'''
#     # If no stop is specified, I think it just goes to the end of the genome sequence. 
#     url = f'https://www.genome.jp/dbget-bin/cut_sequence_genes.pl?FROM=1&ORG={organism}'

#     def is_nucleotides(s:str) -> bool:
#         return re.match('^[ATGC]+$', s) is not None
    
#     def is_header(s:str) -> bool:
#         return '>' in s

#     # Get the genome and parse the returned HTML text string (i.e. remove font color and titles).
#     soup = BeautifulSoup(requests.get(url).text, 'html.parser')
#     parsed = soup.get_text().split('\n')
#     # Remove the title from the HTML file. 
#     parsed = [line for line in parsed if (is_nucleotides(line) or is_header(line))]
    
#     return '\n'.join(parsed)


def get_genomes_by_ko(ko:str, genes_path:str=None, output_path:str='/home/prichter/Documents/selenobot/data/validation/v2/genomes'):
    assert output_path is not None, 'get_genomes_by_ko: A path for the genomes must be specified.'
    if genes_path is not None:
        genes_df = pd.read_csv(genes_path)
    else:
        genes_df = get_metadata_by_ko(ko)
    # Get the organism codes from the DataFrame. 
    organisms = set(genes_df.id.str.split(':', expand=True)[0])
    for organism in organisms:
        if f'{organism}.fn' in os.listdir(output_path):
            print(f'get_genomes_by_ko: Genome of organism {organism} is already downloaded.')
            continue
        print(f'get_genomes_by_ko: Downloading genome for organism {organism}.')
        genome = get_genome(organism)
        with open(os.path.join(output_path, f'{organism}.fn'), 'w') as f:
            f.write(genome)


def get_metadata_by_ko(ko:str, n:int=None):
    '''Retrieves metadata for all genes in a specified KO group.'''
    url = f'https://rest.kegg.jp/get/ko:{ko}'
    text = requests.get(url).text
    assert text is not None, 'get_metadata_by_ko: Something went wrong with HTTP request to KEGG.'
    ko_info = parse_ko_file(text=text)

    df = []
    genes = ko_info['genes'] if (n is None) else ko_info['genes'][:n]
    for gene in tqdm.tqdm(genes, desc=f'get_metadata_by_ko: Retrieving genes from KEGG for KO group {ko}...'):
        url = f'https://rest.kegg.jp/get/{gene}'
        text = requests.get(url).text
        assert text is not None, 'get_metadata_by_ko: Something went wrong with HTTP request to KEGG.'
        row = parse_gene_file(text=text)
        row['id'] = gene
        df.append(row)
    df = pd.DataFrame(df)
    df['ko'] = ko
    return df


def get_metadata_by_genes(ids:List[str]):
    df = []
    for id_ in tqdm.tqdm(ids, desc='get_metadata_by_genes: Retrieving gene data from KEGG'):
        url = f'https://rest.kegg.jp/get/{id_}'
        text = requests.get(url).text
        assert text is not None, 'get_metadata_by_ko: Something went wrong with HTTP request to KEGG.'
        row = parse_gene_file(text=text)  
        row['id'] = id_
        df.append(row)
    return pd.DataFrame(df)    


def check_genome(path:str) -> NoReturn:
    pass
