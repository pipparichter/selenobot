import os 
import re
from typing import Dict, List, NoReturn
import pandas as pd 
import numpy as np
import h5py 
from bs4 import BeautifulSoup, SoupStrainer
from tqdm import tqdm

def get_converter(dtype):
    '''Function for getting type converters to make things easier when reading in the metadata files.'''
    if dtype == str:
        def converter(val):
            return str(val)
    elif dtype == int:
        def converter(val):
            if val == 'none':
                return -1
            else:
                return int(val)
    elif dtype == float:
        def converter(val):
            if val == 'none':
                return -1.0
            else:
                return float(val)
    return converter


class File():

    def __init__(self, path:str):

        if path is not None:
            self.path = path 
            self.dir_name, self.file_name = os.path.split(path) 
        # self.data = None # This will be populated with a DataFrame in child classes. 
        self.genome_id = None # This will be populated with the genome ID extracted from the filename for everything but the MetadataFile class.

    def dataframe(self):
        return self.data


class ClstrFile(File):

    def __init__(self, path:str):

        with open(path, 'r') as f:
            self.content = f.read()

    def clusters(self):
        # The start of each new cluster is marked with a line like ">Cluster [num]"
        return re.split(r'^>.*', self.content, flags=re.MULTILINE)

    def dataframe(self) -> pd.DataFrame:
        '''Convert a ClstrFile to a pandas DataFrame.'''
        df = {'gene_id':[], 'cluster':[]}
        # Split on the newline. 
        for i, cluster in enumerate(self.clusters()):
            pattern = r'>gene_id=([\w\d_\[\]]+)' # Pattern to extract gene ID from line. 
            gene_ids = [re.search(pattern, x).group(1).strip() for x in cluster.split('\n') if x != '']
            df['gene_id'] += gene_ids
            df['cluster'] += [i] * len(gene_ids)

        df = pd.DataFrame(df) # .set_index('id')
        df.cluster = df.cluster.astype(int) # This will speed up grouping clusters later on. 
        return df


# TODO: Should probably use BioPython for this. Was there a reason I didn't?
class FastaFile(File):

    def __init__(self, path:str, content:str=None, genome_id:str=None):
        '''Initialize a FastaFile object.

        :param path: The path to the FASTA file. 
        :param content: Provides the option of initializing a FastaFile directly from file contents. 
        :param genome_id: Provides the option of specifying the genome ID if a file is initialized from contents.'''
          
        super().__init__(path) 

        self.n_entries = None
        if path is not None:
            with open(path, 'r') as f:
                self.content = f.read()
            # self.genome_id = re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name).group(0)
        else:
            self.content = content
        self.genome_id = genome_id


    def parse_header(self, header:str) -> dict:
        return {'id':[h.replace('>', '') for h in headers]}

    def headers(self):
        '''Extract all sequence headers stored in a FASTA file.'''
        return list(re.findall(r'^>.*', self.content, re.MULTILINE))

    def sequences(self):
        '''Extract all  sequences stored in a FASTA file.'''
        seqs = re.split(r'^>.*', self.content, flags=re.MULTILINE)[1:]
        # Strip all of the newline characters from the amino acid sequences. 
        seqs = [s.replace('\n', '') for s in seqs]

        return seqs

    def __len__(self):
        # Avoid re-computing the number of entries each time. 
        if self.n_entries is None:
            self.n_entries = len(self.headers())
        return self.n_entries

    def dataframe(self) -> pd.DataFrame:
        '''Load a FASTA file in as a pandas DataFrame. If the FASTA file is for a particular genome, then 
        add the genome ID as an additional column.'''
        df = [self.parse_header(header) for header in self.headers()]
        for row, seq in zip(df, self.sequences()):
            row['seq'] = seq
        return pd.DataFrame(df)

    def write(self, path:str):

        with open(path, 'w') as f:
            f.write(self.content)


class EmbeddingsFile(File):
    '''Handles reading files containing Prot-T5 embeddings, which are stored on HPC as HDF files'''
    @staticmethod
    def fix_gene_id(gene_id:str) -> str:
        '''Josh replaced all the periods in the gene IDs with underscores, for some reason, so need to get them back into 
        the normal form...'''
        # The keys in the data are the entire Prodigal header string, so need to first get the gene ID out. 
        gene_id = gene_id.split('#')[0].strip()
        if gene_id.count('_') > 2:
            # If there are more than two underscores, then there is an underscore in the main part of the gene ID. 
            # In this case, we want to replace the second underscore. 
            idx = gene_id.find('_') # Get the first occurrence of the underscore. 
            first_part = gene_id[:idx + 1]
            second_part = gene_id[idx + 1:].replace('_', '.', 1) # Replace the underscore in the second part of the ID. 
            return first_part + second_part
        else:
            return gene_id.replace('_', '.', 1)


    def __init__(self, path:str):
        
        super().__init__(path)
        _, self.file_type = os.path.splitext(self.file_name)

        # self.genome_id = None
        # if re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name) is not None:
        #     self.genome_id = re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name).group(0)

        if self.file_type == '.h5':
            data = h5py.File(path, 'r')
            # NOTE: Gene IDs have trailing whitespace, so make sure to remove. 
            # self.gene_ids = [key.split('#')[0].replace('_', '.', 1).strip() for key in data.keys()]
            self.gene_ids = [EmbeddingsFile.fix_gene_id(key) for key in data.keys()]
            # Read in the embeddings from the H5 file, one at a time. Each embedding is stored under a separate key. 
            embeddings = []
            for key in data.keys():
                emb = np.empty(data[key].shape)
                data[key].read_direct(emb)
                embeddings.append(np.expand_dims(emb, axis=0)) # Expand dimensions so concatenation works correctly. 
            self.embeddings = np.concatenate(embeddings, axis=0)
            data.close()

        elif self.file_type == '.csv':
            raise Exception('TODO')
        else:
            raise Exception(f'EmbeddingsFile.__init__: Invalid file type {self.file_type}.')
        

    def keys(self):
        return self.gene_ids
    
    def values(self):
        return list(self.embeddings)

    def items(self):
        return zip(self.keys(), self.values())

    def __len__(self):
        return len(self.gene_ids)

    def dataframe(self):

        df = pd.DataFrame(self.embeddings)
        df['gene_id'] = self.gene_ids
        return df


class NcbiXmlFile(File):
    '''These files are obtained from the NCBI FTP site, and contain metadata information for sequences in SwissProt (files are also available
    for TREMBL entries, but I did not download these).
    
    There is currently no way to get nucleotide sequences or genomes.'''
    # tags = ['taxon', 'accession', 'entry', 'organism', 'sequence']
    # tags = ['accession', 'organism', 'sequence']

    @staticmethod
    def get_taxonomy(organism) -> Dict[str, str]:
        '''Extract the taxonomy information from the organism tag group.'''
        levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        taxonomy = {level:tag.text for tag, level in zip(organism.find_all('taxon'), levels)}
        taxonomy['species'] = organism.find('name').text
        taxonomy['ncbi_taxonomy_id'] = organism.find(name='dbReference', attrs={'type':'NCBI Taxonomy'})['id'] # , attrs={'type':'NCBI Taxonomy'})[0].id
        return taxonomy

    @staticmethod
    def get_refseq(entry) -> Dict[str, str]:
        '''Get references to RefSeq database in case I want to access the nucleotide sequence later on.'''
        refseq = dict()
        refseq_entry = entry.find('dbReference', attrs={'type':'RefSeq'}) # Can we assume there is always a RefSeq entry? No. 
        if refseq_entry:
            refseq['refseq_protein_id'] = refseq_entry['id']
            refseq['refseq_nucleotide_id'] = refseq_entry.find('property', attrs={'type':'nucleotide sequence ID'})['value']
        else:
            refseq['refseq_protein_id'] = None
            refseq['refseq_nucleotide_id'] = None
        return refseq
            
    def __init__(self, path:str, load_seqs:bool=True):
        super().__init__(path)

        with open(path, 'r') as f:
            content = f.read()

        print(f'NcbiXmlFile.__init__: Read in NCBI XML file at path {path}.')
        # strainer = SoupStrainer([tag for tag in NcbiXmlFile.tags if tag != 'sequence']) if (not load_seqs) else SoupStrainer(NcbiXmlFile.tags) 
        soup = BeautifulSoup(content, features='xml') # , parse_only=strainer)

        rows = []
        for entry in tqdm(soup.find_all('entry'), desc='NcbiXmlFile.__init__: Parsing NCBI XML file...'):
            accessions = [tag.text for tag in entry.find_all('accession')] # There may be multiple accessions for the same protein. 
            protein_name = entry.find('name').text # ... But there should only be one name. The protein name is the first "name" tag. 
            
            taxonomy = NcbiXmlFile.get_taxonomy(entry.find('organism'))
            refseq = NcbiXmlFile.get_refseq(entry)

            if load_seqs:
                # Grab the last sequence encountered, which is the latest version, and will actually have the amino acids. 
                seq = entry.find_all('sequence')[-1].text

            # assert len(taxonomy) == len(levels), f'NcbiXmlFile.__init__: There doesn\'t seem to be enough taxonomy data for organism {species}.'
            for accession in accessions:
                row = taxonomy.copy()
                row.update(refseq)
                row['seq'] = seq
                row['name'] = protein_name
                row['gene_id'] = accession
                # row['gene_name'] = gene_name
                rows.append(row)

        self.df = pd.DataFrame(rows).set_index('gene_id')

    def dataframe(self):
        return self.df


class ProteinsFile(FastaFile):

    def __init__(self, path:str, content:str=None, genome_id:str=None):

        super().__init__(path, content=content, genome_id=genome_id)
            
    def parse_header(self, header:str) -> dict:
        '''header strings are of the form >col=val;col=val..., which is the format produced by the 
        dataframe_to_fasta function.
        '''
        # Headers are of the form >col=value;...;col=value
        header = header.replace('>', '') # Remove the header marker. 
        return dict([entry.split('=') for entry in header.split(';')])


    @staticmethod
    def dataframe_to_fasta(df:pd.DataFrame, textwidth:int=80) -> str:
        '''Convert a pandas DataFrame containing FASTA data to a FASTA file format.'''
        if df.index.name in ['gene_id', 'id']:
            df[df.index.name] = df.index

        # Include all non-sequence fields in the FASTA header. 
        header_fields = [col for col in df.columns if col != 'seq']

        fasta = ''
        for row in df.itertuples():
            header = [f'{field}={getattr(row, field)}' for field in header_fields]
            fasta += '>' + ';'.join(header) + '\n' # Add the header to the FASTA file. 

            # Split the sequence up into substrings of length textwidth.
            n = len(row.seq)
            seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]
            assert len(''.join(seq)) == n, 'MyProteinsFile.dataframe_to_fasta: Part of the sequence was lost when splitting into lines.'
            fasta += '\n'.join(seq) + '\n'

        return fasta

    @classmethod
    def from_dataframe(cls, df:pd.DataFrame, genome_id:str=None):

        content = MyProteinsFile.dataframe_to_fasta(df)
        return cls(None, content=content, genome_id=genome_id)


# class NcbiTsvFile(File):
#     '''These files are obtained through the UniProt web interface, either via the API link or directly downloading from the browser.'''
#     fields = {'Entry':'gene_id', 'Reviewed':'reviewed', 'Entry Name':'name', 'Organism':'organism', 'Date of creation':'date', 'Entry version':'version', 'Taxonomic lineage':'taxonomy', 'RefSeq':'refseq_id', 'KEGG':'kegg_id'}
    
#     @staticmethod
#     def parse_taxonomy(df:pd.Series) -> pd.DataFrame:
#         levels = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
#         taxonomy_df = []
#         for entry in df.itertuples():
#             row, taxonomy = {'gene_id':entry.gene_id}, entry.taxonomy
#             for level in range(levels):
#                 if f'({level})' in taxonomy:
#                     row[level] = re.match(f', ([A-Za-z0-9\\w]+) \\({level}\\),', entry).group(1)
#                 else:
#                     row[level] = None
#             taxonomy_df.append(row)
#         taxonomy_df = pd.DataFrame(taxonomy_df).rename(columns={'superkingdom':'domain'})
#         df = df.drop(columns=['taxonomy']).merge(taxonomy_df, how='left', left_on='gene_id', right_on='gene_id')
#         return df

#     def __init__(self, path:str):
#         super().__init__(path)
#         df = pd.read_csv(path, delimiter='\t')
#         df = df.rename(columns=UniProtTsvFile.fields)

#         if 'taxonomy' in df.columns:
#             df = NcbiTsvFile.parse_taxonomy(df)
#         self.df = df.set_index('gene_id')

#     def dataframe(self):
#         return self.df




