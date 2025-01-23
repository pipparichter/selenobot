import os 
import pandas as pd 
import numpy as np
import subprocess
import shutil
import math
from tqdm import tqdm

from selenobot.tools import BLAST
from selenobot.files import BLASTFile, FASTAFile, GBFFFile, fasta_file_parser_gtdb

class Organism():

    def __init__(self, genome_id:str, species:str, dir_:str='../data/model_organisms/'):
        '''Initialize an Organism object.'''

        self.code_name = (species.split()[0][0] + species.split()[-1][:3]).lower() 
        self.dir_ = dir_
        # self.genome_id_prefix = 'RS_' if ('RS_' in genome_id) else 'GB_'
        self.genome_id = genome_id
        self.species = species

        self.genome_path = os.path.join(dir_, f'{self.code_name}_genomic.fna')
        self.gtdb_proteins_path = os.path.join(dir_, f'gtdb_{self.code_name}_protein.faa')
        self.ncbi_proteins_path = os.path.join(dir_,f'ncbi_{self.code_name}_protein.faa')
        self.ncbi_gbff_path = os.path.join(dir_, f'ncbi_{self.code_name}_genomic.gbff')
        self.blast_path = os.path.join(dir_, f'gtdb_{self.code_name}_protein.blast.tsv')

        self.proteins_df = FASTAFile(self.gtdb_proteins_path).to_df(parser=fasta_file_parser_gtdb)
        self.labels = dict()
        self.label_info = dict()

        if os.path.exists(self.ncbi_gbff_path):
            self.gbff_file = GBFFFile(self.ncbi_gbff_path)
        else:
            self.gbff_file = self.download_ncbi_data()

        if os.path.exists(self.blast_path):
            self.blast_file = BLASTFile(self.blast_path)
        else: 
            self.blast_file = self.blast()
    
    def size(self, source:str='gtdb', pseudo:bool=None):
        if source == 'gtdb':
            fasta_file = FASTAFile(self.gtdb_proteins_path) 
            return len(fasta_file)
        elif source == 'ncbi':
            gbff_df = self.gbff_file.to_df(pseudo=pseudo)
            return len(gbff_df)

    def get_match_info(self):

        gbff_df = self.gbff_file.to_df(pseudo=False, drop_duplicates=True)
        match_df = self.label_info['match'].merge(gbff_df, left_on='subject_id', right_on='protein_id', how='left')
        match_df = match_df.drop(columns=['protein_id'])
        match_df.index = match_df.query_id
        match_df['species'] = self.species
        return match_df

    def get_error_info(self):

        gbff_df = self.gbff_file.to_df(pseudo=False, drop_duplicates=True)
        error_df = self.label_info['error'].merge(gbff_df, left_on='subject_id', right_on='protein_id', how='left')
        error_df = error_df.drop(columns=['protein_id'])
        error_df.index = error_df.query_id
        error_df['species'] = self.species
        return error_df

    def get_pseudo_info(self):

        pseudo_df = self.label_info['pseudo']
        pseudo_df['species'] = self.species
        return pseudo_df

    def __eq__(self, code_name:str):
        return self.code_name == code_name

    def to_df(self, max_seq_length:int=2000, label:str=None):
        df = self.proteins_df 
        df['code_name'] = self.code_name
        df['genome_id'] = self.genome_id 
        df['species'] = self.species
        if len(self.labels) > 0:
            df = df.merge(pd.DataFrame({'label':self.labels}).fillna('none'), right_index=True, left_index=True, how='left')
        if (max_seq_length is not None):
            df = df[df.seq.apply(len) < max_seq_length]
        if (label is not None):
            df = df[df.label == label]
        return df

    def download_ncbi_data(self):

        output_path = os.path.join(self.dir_, 'ncbi.zip')
        cmd = f'datasets download genome accession {self.genome_id} --filename {output_path} --include protein,gbff,genome --no-progressbar'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(f'unzip -o {output_path} -d {self.dir_}', shell=True, check=True, stdout=subprocess.DEVNULL)
            
        file_names = ['protein.faa', 'genomic.gbff', '*genomic.fna']
        src_paths = [os.path.join(self.dir_, 'ncbi_dataset/data', self.genome_id, file_name) for file_name in file_names]
        dst_paths = [self.ncbi_proteins_path, self.ncbi_gbff_path, self.genome_path]
        for src_path, dst_path in zip(src_paths, dst_paths):
            subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True)

        shutil.rmtree(os.path.join(self.dir_, 'ncbi_dataset'))
        os.remove(output_path)
        return GBFFFile(self.ncbi_gbff_path)


    def blast(self, max_high_scoring_pairs=1, max_subject_sequences=1):
        '''Run BLAST for the GTDB genome against the corresponding NCBI reference genome.'''
        blast = BLAST()
        blast.run(self.gtdb_proteins_path, self.ncbi_proteins_path, output_path=self.blast_path, max_high_scoring_pairs=max_high_scoring_pairs, max_subject_sequences=max_subject_sequences, make_database=False)
        return BLASTFile(self.blast_path)


    def search_gbff_file(self, df:pd.DataFrame, func, **kwargs):
        '''Look for sequences in the input GBFF file which overlap somewhere in the specified range.'''
        gbff_df = self.gbff_file.to_df(pseudo=kwargs.get('psuedo', None))
        
        def get_top_hit(start:int=None, stop:int=None, strand:int=None):
            overlap = lambda row : len(np.intersect1d(np.arange(start, stop), np.arange(row.start, row.stop)))
            df = gbff_df[gbff_df.strand == strand] if (strand is not None) else gbff_df
            if len(df) == 0:
                return {'overlap':None}
            df = df.copy() # Copy so it doesn't get mad about the slicing. 
            df['overlap'] = df.apply(overlap, axis=1)
            df['percent_overlap'] = 100 * (df.overlap / (stop - start))
            df = df[df['overlap'] > 0]
            df = df.sort_values('overlap', ascending=False)
            return {'overlap':None} if (len(df) == 0) else df.to_dict(orient='records')[0]

        info_df, mask = list(), list()
        for row in tqdm(list(df.itertuples()), desc='search_gbff_file'):
            hit = get_top_hit(start=int(row.start), stop=int(row.stop))
            val, hit = func(hit, start=int(row.start), stop=int(row.stop))
            hit['id'] = row.Index
            info_df.append(hit)
            mask.append(val)
        mask = np.array(mask)
        info_df = pd.DataFrame(info_df).set_index('id')
        info_df = info_df[~info_df.overlap.isnull()] # This will be null if the provided function returned an emptry dictionary. 
        return mask, info_df


    def find_matches(self, df:pd.DataFrame):
        blast_df = self.blast_file.to_df()

        def is_match(hit) -> bool:
            if pd.isnull(hit.subject_id):
                return False
            if hit.sequence_identity < 95:
                return False  
            if not math.isclose(hit.subject_sequence_length, hit.query_sequence_length, abs_tol=5):
                return False
            return True

        df, blast_df = df.align(blast_df, axis=0, join='left')
        mask = blast_df.apply(is_match, axis=1)

        self.label_info['match'] = blast_df.copy()[mask]

        return df[mask], df[~mask]


    def find_errors(self, df:pd.DataFrame, code_name:str='ecol'):
        ''''''
        # NOTE: Seems worth distinguishing between left and right-side boundary errors. 
        blast_df = BLASTFile(self.blast_path).to_df()

        def is_error(hit) -> bool: # Assuming the exact matches have already been filtered out. 
            if pd.isnull(hit.subject_id):
                return False
            if hit.sequence_identity < 95:
                return False
            return True

        df, blast_df = df.align(blast_df, axis=0, join='left')
        mask = blast_df.apply(is_error, axis=1)

        info_df = blast_df.copy()[mask]
        info_df['left_aligned'] = (info_df.query_alignment_start == 1) & (info_df.subject_alignment_start == 1)
        info_df['right_aligned'] = (info_df.query_alignment_end == info_df.query_sequence_length - 1) & (info_df.subject_alignment_end == info_df.subject_sequence_length)
        self.label_info['error'] = info_df

        return df[mask], df[~mask]


    def find_pseudogenes(self, df:pd.DataFrame):
        ''''''
        def is_psuedogene(hit:dict, start:int=None, stop:int=None):
            if hit['overlap'] is None:
                return False, hit
            if (hit['start'] == start) or (hit['stop'] == stop):
                return True, hit
            if hit['percent_overlap'] > 50:
                return True, hit
            return False, dict()

        mask, info_df = self.search_gbff_file(df, is_psuedogene, psuedo=True)
        self.label_info['pseudo'] = info_df   
        return df[mask], df[~mask]


    def find_intergenic(self, df:pd.DataFrame, allowed_overlap:int=30):
        '''A GTDB ORF is intergenic if it either (1) does not overlap with any other nCDS element in the NCBI reference or (2) the overlap with a 
        protein (non-ppseudo) in the reference genome is no greater than the specified margin. 
        https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-15-721
        https://pmc.ncbi.nlm.nih.gov/articles/PMC525685/
        '''

        def is_intergenic(hit:dict, start:int=None, stop:int=None) -> (bool, dict):
            if hit['overlap'] is None:
                return True, hit
            if hit['percent_overlap'] > 50:
                return False, hit
            if hit['overlap'] > allowed_overlap:
                return False, hit
            hit['overlap'] = None # Make sure to mark the hit as invalid by setting the overlap to None. 
            return True, hit

        mask, info_df = self.search_gbff_file(df, is_intergenic, psuedo=False)  
        self.label_info['inter'] = info_df
        return df[mask], df[~mask]


    def label(self):

        label_dfs = dict()
        label_dfs['match'], proteins_df = self.find_matches(self.proteins_df)
        label_dfs['error'], proteins_df = self.find_errors(proteins_df)
        label_dfs['pseudo'], proteins_df = self.find_pseudogenes(proteins_df)
        label_dfs['inter'], proteins_df = self.find_intergenic(proteins_df)

        labels = dict()
        for label, label_df in label_dfs.items():
            labels.update({id_:label for id_ in label_df.index})
            print(f'Organism.label: Found {len(label_df)} sequences in the input genome with the "{label}" label.')
        print(f'Organism.label: Successfully labeled {len(labels)} of the {len(self.proteins_df)} proteins.')
        labels.update({id_:None for id_ in proteins_df.index}) # Account for the remaining sequences. 
        self.labels = labels

