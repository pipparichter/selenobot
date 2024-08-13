import os 
import re
from typing import Dict, List, NoReturn
import pandas as pd 
import numpy as np
import h5py 

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
            gene_ids = [re.search(pattern, x).group(1) for x in cluster.split('\n') if x != '']
            df['gene_id'] += ids
            df['cluster'] += [i] * len(gene_ids)

        df = pd.DataFrame(df) # .set_index('id')
        df.cluster = df.cluster.astype(int) # This will speed up grouping clusters later on. 
        return df



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

    def __init__(self, path:str):
        
        super().__init__(path)
        _, self.file_type = os.path.splitext(self.file_name)

        # self.genome_id = None
        # if re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name) is not None:
        #     self.genome_id = re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name).group(0)

        if self.file_type == '.h5':
            data = h5py.File(path, 'r')
            self.gene_ids = [key.split('#')[0].replace('_', '.', 1) for key in data.keys()] # The keys in the data are the entire Prodigal header string. 
            # Read in the embeddings from the H5 file, one at a time. Each embedding is stored under a separate key. 
            embeddings = []
            for key in data.keys():
                emb = np.empty(data[key].shape)
                data[key].read_direct(emb)
                embeddings.append(emb)
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

                


class NcbiProteinsFile(FastaFile):

    fields = ['start', 'stop', 'gene_id', 'strand']

    def __init__(self, path:str):

        super().__init__(path)
                
    def parse_header(self, header:str) -> dict:
        # Headers in the downloaded file are of the following form... 
        # lcl|NC_000913.3_prot_NP_414542.1_1 [gene=thrL] [locus_tag=b0001] [db_xref=UniProtKB/Swiss-Prot:P0AD86] [protein=thr operon leader peptide] [protein_id=NP_414542.1] [location=190..255] [gbkey=CDS]
        header_df = []
        for header in headers:
            gene_id = re.search(r'gene=([a-zA-Z\d]+)', header) # .group(1)
            gene_id = None if gene_id is None else gene_id.group(1) # Sometimes there are "Untitled" genes. Also skipping these. 
            
            if re.search(r'location=complement\((\d+)\.\.(\d+)\)', header) is not None:
                location = re.search(r'location=complement\((\d+)\.\.(\d+)\)', header)
                start, stop = int(location.group(1)), int(location.group(2))
                strand = '-'
            elif re.search(r'location=(\d+)\.\.(\d+)', header) is not None:
                location = re.search(r'location=(\d+)\.\.(\d+)', header)
                start, stop = int(location.group(1)), int(location.group(2))
                strand = '+' 
            else:
                # This happens with Joins... Don't really know how to handle them, so just passing over them. 
                start, stop, strand = None, None, None
            header_df.append({'start':start, 'stop':stop, 'gene_id':gene_id, 'strand':strand})
        return pd.DataFrame(header_df)


class MyProteinsFile(FastaFile):

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



