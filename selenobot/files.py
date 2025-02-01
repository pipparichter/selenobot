import os 
import re
from typing import Dict, List, NoReturn
import pandas as pd 
import numpy as np
import h5py 
from typing import List, Dict, Tuple
# from bs4 import BeautifulSoup, SoupStrainer
from lxml import etree
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess


def count_lines(path:str) -> int:
    cmd = f'cat {path} | wc -l'
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    try:
        return int(result.stdout)
    except TypeError:
        raise TypeError(f'count_lines: The function did not return an integer. Returned {result.stdout}')


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


def fasta_file_parser_gtdb(description:str):
    pattern = r'# ([\d]+) # ([\d]+) # ([-1]+) # ID=([^;]+);partial=([^;]+);start_type=([^;]+);rbs_motif=([^;]+);rbs_spacer=([^;]+);gc_cont=([\.\w]+)'
    columns = ['start', 'stop', 'strand', 'ID', 'partial', 'start_type', 'rbs_motif', 'rbs_spacer', 'gc_content']
    match = re.search(pattern, description)
    parsed_header = {col:match.group(i + 1) for i, col in enumerate(columns)}
    parsed_header['contig_number'] = int(parsed_header['ID'].split('_')[0])
    return parsed_header


def fasta_file_parser_none(description:str):
    return {'description':description}


def fasta_file_parser(description:str) -> dict:
    '''Descriptions should be of the form >col=val;col=val. '''
    # Headers are of the form >col=value;...;col=value
    return dict([entry.split('=') for entry in description.split(';')])
        

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
            
    def to_df(self, parser=fasta_file_parser_none) -> pd.DataFrame:
        '''Load a FASTA file in as a pandas DataFrame. If the FASTA file is for a particular genome, then 
        add the genome ID as an additional column.'''

        df = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            row = parser(description)
            row['id'] = id_
            row['seq'] = seq.replace('*', '')
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
    def load(path:str, n_lines:int=None) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter='\t', names=BLASTFile.fields, header=None)
        return df

    @staticmethod
    def load_chunks(path:str, chunk_size:int=500, n_lines:int=None) -> pd.DataFrame:
        
        chunk_dfs = pd.read_csv(path, delimiter='\t', names=BLASTFile.fields, header=None, chunksize=chunk_size)
        n_chunks = int(np.ceil(n_lines / chunk_size))
        
        df = []
        pbar = tqdm(total=n_chunks, desc='BLASTFile.load_chunks: Loading BLAST output in batches...')
        for chunk_df in chunk_dfs:
            df.append(chunk_df)
            pbar.update(1)
        df = pd.concat(df)
        pbar.close()
        return df 

    def __init__(self, path:str):
        '''Initialize a BLASTFile object.

        :param path: The path to the BLAST output file, which is in TSV format. 
        :param max_e_value. For more notes on E-values, see https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/_E_value.html.
        '''

        n_lines = count_lines(path)
        if n_lines > 10000:
            self.df = BLASTFile.load_chunks(path, n_lines=n_lines)
        else:
            self.df = BLASTFile.load(path, n_lines=n_lines)

        self.df = self.df.rename(columns=BLASTFile.field_map) # Rename the columns to more useful things. 
        self.df['id'] = self.df.query_id # Use the query ID as the main ID. 
        self.df = self.df.set_index('id')
        self.df['subject_id'] = self.df.subject_id.apply(lambda id_ : BLASTFile.remove_swissprot_tag(id_))

    # NOTE: There can be alignments for different portions of the query and target sequences, which this does not account for. 
    # I am counting on the fact that this will not effect things much. 

    def drop_duplicate_hsps(self, col:str='alignment_length', how:str='highest'):
        '''A BLAST result can have multiple alignments for the same query-subject pair (each of these alignments is called
        an HSP). If you specify that you only want one HSP per query-target pair, BLAST will just select the alignment with
        the lowest E-value. However, I care more about sequence identity, so I am selecting best HSPs manually.'''
        # There are two relevant parameters per HSP: sequence_identity and length (the length of the aligned region)
        self.df = self.df.sort_values(col, ascending=True if (how == 'lowest') else False)
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


class GBFFFile(File):
    '''Class for parsing GenBank flat files, obtained from NCBI.'''

    fields = ['feature', 'contig_number', 'strand', 'start', 'stop', 'partial', 'product', 'frameshifted', 'incomplete', 'internal_stop', 'protein_id', 'seq', 'pseudo']
    # TODO: I should automatically detect the features...
    features = ['gene', 'CDS', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA','repeat_region', 'misc_feature', 'mobile_element']

    field_pattern = re.compile(r'/([a-zA-Z_]+)="([^"]+)"')
    coordinate_pattern = re.compile(r'complement\([\<\d]+\.\.[\>\d]+\)|[\<\d]+\.\.[\>\d]+')
    feature_pattern = r'[\s]{2,}(' + '|'.join(features) + r')[\s]{2,}'


    @staticmethod
    def parse_coordinate(coordinate:str):
        '''Parse a string indicating gene boundaries. These strings contain information about the start codon location, 
        stop codon location, and strand.'''
        parsed_coordinate = dict()
        parsed_coordinate['strand'] = -1 if ('complement' in coordinate) else 1
        # NOTE: Details about coordinate format: https://www.ncbi.nlm.nih.gov/genbank/samplerecord/
        start, stop = re.findall(r'[\>\<0-9]+', coordinate)
        partial = ('1' if ('<' in start) else '0') + ('1' if ('>' in stop) else '0')
        start = int(start.replace('<', ''))
        stop = int(stop.replace('>', ''))

        parsed_coordinate['start'] = start
        parsed_coordinate['stop'] = stop
        parsed_coordinate['partial'] = partial

        return parsed_coordinate

    @staticmethod
    def parse_note(note:str):
        parsed_note = dict()
        parsed_note['frameshifted'] = 'frameshifted' in note
        parsed_note['internal_stop'] = 'internal stop' in note
        parsed_note['incomplete'] = 'incomplete' in note
        return parsed_note 


    # NOTE: PGAP annotations can include pogrammed frameshift sequences (is this where the duplicates come in?)
    @staticmethod
    def parse_entry(feature:str, entry:str) -> dict:

        # Extract the gene coordinates, which do not follow the typical field pattern. 
        coordinate = re.search(GBFFFile.coordinate_pattern, entry).group(0)
        pseudo = ('/pseudo' in entry)
        entry = re.sub(GBFFFile.coordinate_pattern, '', entry)

        entry = re.sub(r'[\s]{2,}|\n', '', entry) # Remove all newlines or any more than one consecutive whitespace character.

        entry = re.findall(GBFFFile.field_pattern, entry) # Returns a list of matches. 
        parsed_entry = {'feature':feature}
        # Need to account for the fact that a single entry can have multiple GO functions, processes, components, etc.
        for field, value in entry:
            if (field not in parsed_entry):
                parsed_entry[field] = value
            else:
                parsed_entry[field] += ', ' + value
        parsed_entry['coordinate'] = coordinate
        parsed_entry['pseudo'] = pseudo
        parsed_entry.update(GBFFFile.parse_coordinate(coordinate))
        if 'note' in parsed_entry:
            parsed_entry.update(GBFFFile.parse_note(parsed_entry['note']))
       
        return parsed_entry 

    @staticmethod
    def parse_contig(contig:str) -> pd.DataFrame:
        # Because I use parentheses around the features in the pattern, this is a "capturing group", and the label is contained in the output. 
        contig = re.split(GBFFFile.feature_pattern, contig, flags=re.MULTILINE)
        contig_metadata, contig = contig[0], contig[1:] # The first entry in the content is contig metadata. 
        contig_id = re.search(r'VERSION[\s]+([^\n]+)', contig_metadata).group(1)

        if len(contig) == 0:
            return contig_id, None

        entries = [(contig[i], contig[i + 1]) for i in range(0, len(contig), 2)]
        assert np.all(np.isin([entry[0] for entry in entries], GBFFFile.features)), 'GBFFFile.__init__: Something went wrong while parsing the file.'
        entries = [entry for entry in entries if entry[0] != 'gene'] # Remove the gene entries, which I think are kind of redundant with the products. 

        df = []
        for entry in entries:
            df.append(GBFFFile.parse_entry(*entry))
        df = pd.DataFrame(df)
        df = df.rename(columns={col:col.lower() for col in df.columns}) # Make sure all column names are lower-case.
        df = df.rename(columns={'translation':'seq'}) 
        df.index = df.locus_tag
        df.index.name = 'locus_tag' # Need to use the locus tag as the index, as not everything has a protein ID. 
        df['contig_id'] = contig_id

        return contig_id, df 

    def __init__(self, path:str):
        
        with open(path, 'r') as f:
            content = f.read()

        # If there are multiple contigs in the file, the set of features corresponding to the contig is marked by 
        # a "contig" feature.
        # NOTE: I used the lookahead match here because it is not treated as a capturing group. 
        contigs = re.findall(r'LOCUS(.*?)(?=LOCUS|$)', content, flags=re.DOTALL) # DOTALL flag means the dot character also matches newlines.

        self.df, self.contig_ids = [], []
        for i, contig in enumerate(contigs):
            contig_id, df = GBFFFile.parse_contig(contig)
            self.contig_ids.append(contig_id)
            if (df is not None):
                df['contig_number'] = i + 1
                self.df.append(df)
        self.df = pd.concat(self.df)
        self.n_contigs = len(contigs)


    @staticmethod
    def drop_duplicates(df:pd.DataFrame):
        '''Some proteins have duplicate entries, I think because multiple copies of proteins are present in some genomes. This 
        function removes the duplicates, and flags them as having multiple copies.'''
        pseudo_df = df.copy()[df.pseudo] # Keep the pseudogenes to add back in later. 
        df = df.copy()[~df.pseudo] # Remove the pseudogenes, which don't have protein IDs. 

        copy_numbers = df.protein_id.value_counts().rename('copy_number')
        duplicate_protein_ids = df.protein_id[df.protein_id.duplicated()]
        # Double check to make sure all sequences are equal before removing. 
        for protein_id, protein_df in df[df.protein_id.isin(duplicate_protein_ids)].groupby('protein_id'): 
            assert (protein_df.seq.nunique() == 1), f'GBFFFile.remove_duplicates: Not all sequences with protein ID {protein_id} are equal.'
        
        # print(f'GBFFFile.remove_duplicates: Removing duplicate entries for {len(duplicate_protein_ids)} sequences from the GBFF file.')
        df = df.drop_duplicates(subset='protein_id')
        df = df.merge(copy_numbers, left_on='protein_id', right_index=True)
        df = pd.concat([df, pseudo_df]) # Add the pseudogenes back in. 
        return df


    def to_df(self, drop_duplicates:bool=False, **filters):
        df = self.df[GBFFFile.fields]
        if drop_duplicates:
            df = GBFFFile.drop_duplicates(df)
        for col, val in filters.items():
            df = df[df[col] == val]

        return df.dropna(axis=1, how='all')



