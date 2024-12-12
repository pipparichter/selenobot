import os 
import re
from typing import Dict, List, NoReturn
import pandas as pd 
import numpy as np
import h5py 
# from bs4 import BeautifulSoup, SoupStrainer
from lxml import etree
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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



class CdhitClstrFile(File):

    def __init__(self, path:str):
        '''
        An example cluster entry is given below:
        >Cluster 4
        0	6aa, >A0A2E7JSV1[1]... at 83.33%
        1	6aa, >A0A6B1IEQ5[1]... at 83.33%
        2	6aa, >A0A6L7QK86[1]... at 83.33%
        3	6aa, >A0A6L7QK86[1]... *
        '''

        with open(path, 'r') as f:        
            content = f.read()

        # The start of each new cluster is marked with a line like ">Cluster [num]"
        content = re.split(r'^>.*', content, flags=re.MULTILINE)
        self.ids, self.clusters, self.similarities = [], [], []
        for i, cluster in enumerate(content):
            entries = [entry for entry in cluster.split('\n') if (entry != '')] # Get all entries in the cluster. 
            ids = [re.search(r'>([\w\d_\[\]]+)', entry).group(1).strip() for entry in entries]
            # similarities = [None if ('*' in entry) else re.search(r"\.\.\. at (\d+\.\d+)\%", entry).group(1).strip() for entry in entries]
            self.ids += ids
            # self.similarities += similarities
            self.clusters += [i] * len(ids) 

    def to_df(self) -> pd.DataFrame:
        '''Convert a ClstrFile to a pandas DataFrame.'''
        df = pd.DataFrame({'id':self.ids, 'cluster':self.clusters}) # , 'similarity':self.similarities}) # .set_index('id')
        df.cluster = df.cluster.astype(int) # This will speed up grouping clusters later on. 
        df.similarity = df.cluster.astype(float) # This will speed up grouping clusters later on. 
        return df.set_index('id')


# TODO: Should probably use BioPython for this. Was there a reason I didn't?
class FastaFile(File):

    def __init__(self, path:str=None, seqs:List[str]=None, ids:List[str]=None, descriptions:List[str]=None):
        '''Initialize a FastaFile object.'''
        super().__init__(path) 

        if (path is not None):
            f = open(path, 'r')
            self.seqs, self.ids, self.descriptions = [], [], []
            for record in SeqIO.parse(path, 'fasta'):
                self.ids.append(record.id)
                self.descriptions.append(record.description.replace(record.id, '').strip())
                self.seqs.append(str(record.seq))
            f.close()
        else:
            self.seqs, self.ids, self.descriptions = seqs, ids, descriptions

    def __len__(self):
        return len(self.seqs)

    @classmethod
    def from_df(cls, df:pd.DataFrame, include_cols:List[str]=None):
        ids = df.index.values.tolist()
        seqs = df.seq.values.tolist()

        include_cols = df.columns if (include_cols is None) else include_cols
        cols = [col for col in df.columns if (col != 'seq') and (col in include_cols)]
        descriptions = []
        for row in df[include_cols].itertuples():
            description = ';'.join([f'{col}={getattr(row, col, None)}' for col in include_cols if (getattr(row, col, None) is not None)])
            descriptions.append(description)
        return cls(ids=ids, seqs=seqs, descriptions=descriptions)
            
    def to_df(self, parse_description:bool=True) -> pd.DataFrame:
        '''Load a FASTA file in as a pandas DataFrame. If the FASTA file is for a particular genome, then 
        add the genome ID as an additional column.'''

        def parse(description:str) -> dict:
            '''Descriptions should be of the form >col=val;col=val. '''
            # Headers are of the form >col=value;...;col=value
            return dict([entry.split('=') for entry in description.split(';')])
        
        df = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            row = {'description':description} if (not parse_description) else parse(description)
            row['id'] = id_
            row['seq'] = seq 
            df.append(row)

        return pd.DataFrame(df).set_index('id')

    def write(self, path:str) -> NoReturn:
        f = open(path, 'w')
        records = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            record = SeqRecord(Seq(seq), id=id_, description=description)
            records.append(record)
        SeqIO.write(records, f, 'fasta')
        f.close()



class EmbeddingsFile(File):
    '''Handles reading files containing Prot-T5 embeddings, which are stored on HPC as HDF files'''
    @staticmethod
    def fix_id(id:str) -> str:
        '''Josh replaced all the periods in the gene IDs with underscores, for some reason, so need to get them back into 
        the normal form...'''
        # The keys in the data are the entire Prodigal header string, so need to first get the gene ID out. 
        id_ = id_.split('#')[0].strip()
        if id_.count('_') > 2:
            # If there are more than two underscores, then there is an underscore in the main part of the gene ID. 
            # In this case, we want to replace the second underscore. 
            idx = id_.find('_') # Get the first occurrence of the underscore. 
            first_part = id_[:idx + 1]
            second_part = id_[idx + 1:].replace('_', '.', 1) # Replace the underscore in the second part of the ID. 
            return first_part + second_part
        else:
            return id_.replace('_', '.', 1)


    def __init__(self, path:str):
        
        super().__init__(path)
        _, self.file_type = os.path.splitext(self.file_name)

        # self.genome_id = None
        # if re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name) is not None:
        #     self.genome_id = re.search(r'GC[AF]_\d{9}\.\d{1}', self.file_name).group(0)

        if self.file_type == '.h5':
            data = h5py.File(path, 'r')
            # NOTE: Gene IDs have trailing whitespace, so make sure to remove. 
            # self.ids = [key.split('#')[0].replace('_', '.', 1).strip() for key in data.keys()]
            self.ids = [EmbeddingsFile.fix_id(key) for key in data.keys()]
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
        return self.ids
    
    def values(self):
        return list(self.embeddings)

    def items(self):
        return zip(self.keys(), self.values())

    def __len__(self):
        return len(self.ids)

    def to_df(self):
        df = pd.DataFrame(self.embeddings)
        df['id'] = self.ids
        return df


class NcbiXmlFile(File):
    '''These files are obtained from the NCBI FTP site, and contain metadata information for sequences in SwissProt (files are also available
    for TREMBL entries, but I did not download these).'''
    # tags = ['taxon', 'accession', 'entry', 'organism', 'sequence']
    # tags = ['accession', 'organism', 'sequence']

    @staticmethod
    def find(namespace:str, elem, name:str, attrs:Dict[str, str]=None):
        xpath = f'.//{namespace}{name}'
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.find(xpath)

    @staticmethod
    def findall(namespace:str, elem, name:str, attrs:Dict[str, str]=None):
        xpath = f'.//{namespace}{name}'
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.findall(xpath)

    @staticmethod
    def get_tag(elem) -> str:
        namespace, tag = elem.tag.split('}') # Remove the namespace from the tag. 
        namespace = namespace + '}'
        return namespace, tag 

    @staticmethod
    def get_taxonomy(namespace:str, entry) -> Dict[str, str]:
        '''Extract the taxonomy information from the organism tag group.'''
        levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        taxonomy = {level:taxon.text for taxon, level in zip(NcbiXmlFile.findall(namespace, entry, 'taxon'), levels)}
        taxonomy['species'] = NcbiXmlFile.find(namespace, entry, 'name').text
        taxonomy['ncbi_taxonomy_id'] = NcbiXmlFile.find(namespace, entry, 'dbReference', attrs={'type':'NCBI Taxonomy'}).attrib['id'] # , attrs={'type':'NCBI Taxonomy'})[0].id
        return taxonomy

    @staticmethod
    def get_refseq(namespace:str, entry) -> Dict[str, str]:
        '''Get references to RefSeq database in case I want to access the nucleotide sequence later on.'''
        refseq = dict()
        refseq_entry = NcbiXmlFile.find(namespace, entry, 'dbReference', attrs={'type':'RefSeq'}) # Can we assume there is always a RefSeq entry? No. 
        if (refseq_entry is not None):
            refseq['refseq_protein_id'] = refseq_entry.attrib['id']
            refseq['refseq_nucleotide_id'] = NcbiXmlFile.find(namespace, refseq_entry, 'property', attrs={'type':'nucleotide sequence ID'}).attrib['value']
        else:
            refseq['refseq_protein_id'] = None
            refseq['refseq_nucleotide_id'] = None
        return refseq
            
    def __init__(self, path:str, load_seqs:bool=True, chunk_size:int=100):
        super().__init__(path)

        pbar = tqdm(etree.iterparse(path, events=('start', 'end')), desc='NcbiXmlFile.__init__: Parsing NCBI XML file...')
        entry, df = None, []
        for event, elem in pbar:
            namespace, tag = NcbiXmlFile.get_tag(elem)
            if (tag == 'entry') and (event == 'start'):
                entry = elem
            if (tag == 'entry') and (event == 'end'):
                accessions = [accession.text for accession in entry.findall(namespace + 'accession')]
                row = NcbiXmlFile.get_taxonomy(namespace, entry) 
                row.update(NcbiXmlFile.get_refseq(namespace, entry))

                if load_seqs:
                    row['seq'] = NcbiXmlFile.findall(namespace, entry, 'sequence')[-1].text
                row['name'] = NcbiXmlFile.find(namespace, entry, 'name').text 
                
                for accession in accessions:
                    row['id'] = accession 
                    df.append(row.copy())
                elem.clear()
                pbar.update(len(accessions))
                pbar.set_description(f'NcbiXmlFile.__init__: Parsing NCBI XML file, row {len(df)}...')

        self.df = pd.DataFrame(df).set_index('id')


    def to_df(self):
        return self.df

    def fasta(self, path:str) -> NoReturn:
        pass




# class NcbiTsvFile(File):
#     '''These files are obtained through the UniProt web interface, either via the API link or directly downloading from the browser.'''
#     fields = {'Entry':'id', 'Reviewed':'reviewed', 'Entry Name':'name', 'Organism':'organism', 'Date of creation':'date', 'Entry version':'version', 'Taxonomic lineage':'taxonomy', 'RefSeq':'refseq_id', 'KEGG':'kegg_id'}
    
#     @staticmethod
#     def parse_taxonomy(df:pd.Series) -> pd.DataFrame:
#         levels = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
#         taxonomy_df = []
#         for entry in df.itertuples():
#             row, taxonomy = {'id':entry.id}, entry.taxonomy
#             for level in range(levels):
#                 if f'({level})' in taxonomy:
#                     row[level] = re.match(f', ([A-Za-z0-9\\w]+) \\({level}\\),', entry).group(1)
#                 else:
#                     row[level] = None
#             taxonomy_df.append(row)
#         taxonomy_df = pd.DataFrame(taxonomy_df).rename(columns={'superkingdom':'domain'})
#         df = df.drop(columns=['taxonomy']).merge(taxonomy_df, how='left', left_on='id', right_on='id')
#         return df

#     def __init__(self, path:str):
#         super().__init__(path)
#         df = pd.read_csv(path, delimiter='\t')
#         df = df.rename(columns=UniProtTsvFile.fields)

#         if 'taxonomy' in df.columns:
#             df = NcbiTsvFile.parse_taxonomy(df)
#         self.df = df.set_index('id')

#     def to_df()(self):
#         return self.df




