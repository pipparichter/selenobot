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


# TODO: Do I need to be able to write things back to FASTA files? I think yes. For CD-HIT clustering, will need to 
# dereplicate first and then group to separate for the train, test, validation split. 

# TODO: Should probably use BioPython for this. Was there a reason I didn't?
class FASTAFile(File):

    def __init__(self, path:str=None, seqs:List[str]=None, ids:List[str]=None, descriptions:List[str]=None):
        '''Initialize a FASTAFile object.'''
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
    def from_df(cls, df:pd.DataFrame, add_description:bool=True):
        ids = df.index.values.tolist() # Expects the index to contain the IDs. 
        seqs = df.seq.values.tolist()
        descriptions = [''] * len(seqs)

        if add_description:
            cols = [col for col in df.columns if (col != 'seq')]
            descriptions = []
            for row in df.drop(columns=['seq']).itertuples():
                # Sometimes there are equal signs in the descriptions, which mess everything up... 
                description = {col:getattr(row, col) for col in cols if (getattr(row, col, None) is not None)}
                description = {col:value.replace('=', '').strip() for col, value in description.items() if (type(value) == str)}
                description = ';'.join([f'{col}={value}' for col, value in description.items()])
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


class MMseqsFile(File):
    def __init__(self, path:str):
        '''Manages the parsing of an MMseqs cluster file. The output is a TSV with two columns, where the first column is the 
        cluster representative and the second column is the cluster member.'''
        self.df = pd.read_csv(path, delimiter='\t', names=['mmseqs_representative', 'id'])
        # Add integer IDs for each cluster. 
        cluster_ids = {rep:i for i, rep in enumerate(self.df.mmseqs_representative.unique())}
        self.df['mmseqs_cluster'] = [cluster_ids[rep] for rep in self.df.mmseqs_representative]

    def to_df(self, reps_only:bool=False) -> pd.DataFrame:
        if reps_only:
            df = self.df.drop_duplicates('mmseqs_representative', keep='first').set_index('id')
        else:
            df = self.df.set_index('id')
        return df


class BLASTFile(File):
    '''Manages the parsing of an BLAST output file in tabular output format 6. See the following link
    for more information on the format: https://www.metagenomics.wiki/tools/blast/blastn-output-format-6'''

    fields = ['qseqid', 'sseqid','pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'] # This is the default. 
    fields += ['qcovs', 'qcovhsp', 'qlen', 'slen'] # Some extra stuff which is helpful. 

    field_map = dict()
    field_map['qseqid'] = 'query_id' # Query or source (gene) sequence id
    field_map['sseqid'] = 'subject_id' # Subject or target (gene) sequence id
    field_map['pident'] = 'sequence_identity' # Percentage of identical positions
    field_map['length'] = 'alignment_length' # Alignment length (sequence overlap)
    field_map['mismatch'] = 'mismatch' # Number of mismatches
    field_map['gapopen'] = 'num_gap_openings' # The number of gap openings. 
    # NOTE: Are the start and end indices zero- or one-indexed?
    field_map['qstart'] = 'query_alignment_start' # Start of alignment in query.
    field_map['qend'] = 'query_alignment_end' # End of alignment in query. 
    field_map['sstart'] = 'subject_alignment_start' # Start of alignment in subject. 
    field_map['send'] = 'subject_alignment_end' # End of alignment in subject. 
    field_map['evalue'] = 'e_value' # E-value https://www.metagenomics.wiki/tools/blast/evalue
    field_map['bitscore'] = 'bit_score' # Bit-score https://www.metagenomics.wiki/tools/blast/evalue 
    field_map['qlen'] = 'query_sequence_length' 
    field_map['slen'] = 'subject_sequence_length' 
    # https://www.biostars.org/p/121972/
    field_map['qcovs'] = 'query_coverage_per_subject' 
    field_map['qcovhsp'] = 'query_coverage_per_pair' # This seems to be a percentage, equal to alignment_length / query_sequence_length.

    @staticmethod
    def remove_swissprot_tag(id_:str):
        '''I am not sure why the BLAST tool has started doing this, but it is appending a sp|{id_}| tag to some
        of the subject sequences, which is not present in the original FASTA file.'''
        match = re.match(r'sp\|([A-Za-z0-9]+)\|', id_)
        return match.group(1) if (match is not None) else id_

    @staticmethod
    def adjust_sequence_identity(row):

        adjusted_sequence_identity = (row.alignment_length - row.mismatch)
        adjusted_sequence_identity = adjusted_sequence_identity / max(row.query_sequence_length, row.subject_sequence_length)
        return adjusted_sequence_identity

    def __init__(self, path:str):

        self.df = pd.read_csv(path, delimiter='\t', names=BLASTFile.fields)
        self.df = self.df.rename(columns=BLASTFile.field_map) # Rename the columns to more useful things. 
        self.df['id'] = self.df.query_id # Use the query ID as the main ID. 
        self.df = self.df.set_index('id')
        
        self.df['subject_id'] = self.df.subject_id.apply(lambda id_ : BLASTFile.remove_swissprot_tag(id_))
        # Adjust the sequence identity to account for alignment length. 
        self.df['adjusted_sequence_identity'] = df.apply(BLASTFile.adjust_sequence_identity, axis=1)

    # NOTE: There can be alignments for different portions of the query and target sequences, which this does not account for. 
    # I am counting on the fact that this will not effect things much. 

    def drop_duplicate_hsps(self, col:str='adjusted_sequence_identity', how:str='highest'):
        '''A BLAST result can have multiple alignments for the same query-subject pair (each of these alignments is called
        an HSP). If you specify that you only want one HSP per query-target pair, BLAST will just select the alignment with
        the lowest E-value. However, I care more about sequence identity, so I am selecting best HSPs manually.'''

        # There are two relevant parameters per HSP: sequence_identity and length (the length of the aligned region)
        if how == 'lowest':
            ascending = True 
        elif how == 'highest':
            ascending = False
        else:
            print('BLASTFile: Specified selection method must be one of \'highest\', \'lowest\'.')
        
        self.df = self.df.sort_values(col, ascending=ascending)
        self.df.drop_duplicates(subset=['query_id', 'subject_id'], keep='first', inplace=True)
   
    def to_df(self) -> pd.DataFrame:
        return self.df


# TODO: Are the amino acids sequences listed in each cluster in any particular order?

class CDHITFile(File):

    def __init__(self, path:str):
        '''
        An example cluster entry is given below:

        >Cluster 4
        0	6aa, >A0A2E7JSV1[1]... at 83.33%
        1	6aa, >A0A6B1IEQ5[1]... at 83.33%
        2	6aa, >A0A6L7QK86[1]... at 83.33%
        3	6aa, >A0A6L7QK86[1]... *

        The asterisk marks the representative sequence of the cluster. As described in the user guide, this is
        simply the longest sequence in the cluster. 
        '''

        with open(path, 'r') as f:        
            content = f.read()

        # The start of each new cluster is marked with a line like ">Cluster [num]"
        content = re.split(r'^>.*', content, flags=re.MULTILINE)

        self.n_clusters = len(content)
        self.cluster_sizes = []

        self.ids, self.clusters, self.representative = [], [], []

        for i, cluster in enumerate(content):
            entries = [entry.strip() for entry in cluster.split('\n') if (entry != '')] # Get all entries in the cluster. 
            ids = [re.search(r'>([\w\d_\[\]]+)', entry).group(1).strip() for entry in entries]
            self.ids += ids 
            self.representative += [True if (entry[-1] == '*') else False for entry in entries] # TODO: Make sure there is one per cluster. 
            self.clusters += [i] * len(ids) 
            self.cluster_sizes.append(len(entries))

    def to_df(self, reps_only:bool=False) -> pd.DataFrame:
        '''Convert a CDHITFile to a pandas DataFrame.'''
        df = pd.DataFrame({'id':self.ids, 'cdhit_cluster':self.clusters, 'cdhit_representative':self.representative}) 
        df.cdhit_cluster = df.cdhit_cluster.astype(int) # This will speed up grouping clusters later on. 
        if reps_only: # If specified, only get the representatives from each cluster. 
            df = df[df.cdhit_representative]
        return df.set_index('id')


class XMLFile(File):
    # tags = ['taxon', 'accession', 'entry', 'organism', 'sequence']
    # tags = ['accession', 'organism', 'sequence']
    # TODO: Read more about namespaces. 


    def find(self, elem, name:str, attrs:Dict[str, str]=None):
        '''Find the first tag in the entry element which has the specified names and attributes.'''
        xpath = f'.//{self.namespace}{name}' #TODO: Remind myself how these paths work. 
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.find(xpath)


    def findall(self, elem, name:str, attrs:Dict[str, str]=None):
        '''Find all tags in the entry element which have the specified names and attributes.'''
        xpath = f'.//{self.namespace}{name}'
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.findall(xpath)

    @staticmethod
    def get_tag(elem) -> str:
        # Namespaces look like [EXAMPLE] specify the location in the tree. 
        namespace, tag = elem.tag.split('}') # Remove the namespace from the tag. 
        namespace = namespace + '}'
        return namespace, tag 

    def get_annotation(self, entry) -> Dict[str, str]:
        '''Grab the functional description and KEGG ortho group (if they exist) for the entry.'''
        annotation = dict()
        kegg_entry = self.find(entry, 'dbReference', attrs={'type':'KEGG'}) 
        if kegg_entry is not None:
            annotation['kegg'] = kegg_entry.attrib['id']
        function_entry = self.find(entry, 'comment', attrs={'type':'function'}) 
        if function_entry is not None:
            # Need to look at the "text" tag stored under the function entry.
            annotation['function'] = self.find(function_entry, 'text').text 
        return annotation

    def get_taxonomy(self, entry) -> Dict[str, str]:
        '''Extract the taxonomy information from the organism tag group.'''
        levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        taxonomy = {level:taxon.text for taxon, level in zip(self.findall(entry, 'taxon'), levels)}
        taxonomy['species'] = self.find(entry, 'name').text
        taxonomy['ncbi_taxonomy_id'] = self.find(entry, 'dbReference', attrs={'type':'NCBI Taxonomy'}).attrib['id'] # , attrs={'type':'NCBI Taxonomy'})[0].id
        return taxonomy

    def get_refseq(self, entry) -> Dict[str, str]:
        '''Get references to RefSeq database in case I want to access the nucleotide sequence later on.'''
        refseq = dict()
        refseq_entry = self.find(entry, 'dbReference', attrs={'type':'RefSeq'}) # Can we assume there is always a RefSeq entry? No. 
        if (refseq_entry is not None):
            refseq['refseq_protein_id'] = refseq_entry.attrib['id']
            refseq['refseq_nucleotide_id'] = self.find(refseq_entry, 'property', attrs={'type':'nucleotide sequence ID'}).attrib['value']
        else:
            refseq['refseq_protein_id'] = None
            refseq['refseq_nucleotide_id'] = None
        return refseq

    def get_non_terminal_residue(self, entry) -> Dict[str, str]:
        '''If the entry passed into the function has a non-terminal residue(s), find the position(s) where it occurs; 
        there can be two non-terminal residues, one at the start of the sequence, and one at the end.'''
        # Figure out of the sequence is a fragment, i.e. if it has a non-terminal residue. 
        non_terminal_residue_entries = self.findall(entry, 'feature', attrs={'type':'non-terminal residue'})
        # assert len(non_terminal_residues) < 2, f'XMLFile.__init__: Found more than one ({len(non_terminal_residue)}) non-terminal residue, which is unexpected.'
        if len(non_terminal_residue_entries) > 0:
            positions = []
            for non_terminal_residue_entry in non_terminal_residue_entries:
                # Get the location of the non-terminal residue. 
                position = self.find(non_terminal_residue_entry, 'position').attrib['position']
                positions.append(position)
            positions = ','.join(positions)
        else:
            positions = None
        return {'non_terminal_residue':positions}
                    
    def __init__(self, path:str, load_seqs:bool=True, chunk_size:int=100):
        super().__init__(path)

        pbar = tqdm(etree.iterparse(path, events=('start', 'end')), desc='XMLFile.__init__: Parsing NCBI XML file...')
        entry, df = None, []
        for event, elem in pbar: # The file tree gets accumulated in the elem variable as the iterator progresses. 
            namespace, tag = XMLFile.get_tag(elem) # Extract the tag and namespace from the element. 
            self.namespace = namespace # Save the current namespace in the object.

            if (tag == 'entry') and (event == 'start'):
                entry = elem
            if (tag == 'entry') and (event == 'end'):
                accessions = [accession.text for accession in entry.findall(namespace + 'accession')]
                row = self.get_taxonomy(entry) 
                row.update(self.get_refseq(entry))
                row.update(self.get_non_terminal_residue(entry))
                row.update(self.get_annotation(entry))

                if load_seqs:
                    row['seq'] = self.findall(entry, 'sequence')[-1].text
                row['name'] = self.find(entry, 'name').text 

                for accession in accessions:
                    row['id'] = accession 
                    df.append(row.copy())

                elem.clear() # Clear the element to avoid loading everything into memory. 
                pbar.update(len(accessions))
                pbar.set_description(f'XMLFile.__init__: Parsing NCBI XML file, row {len(df)}...')

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




