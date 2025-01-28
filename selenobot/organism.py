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

        self.gbff_search_results = dict()
    
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

    @staticmethod
    def is_valid_hit(hit:dict, start:int=None, stop:int=None, strand:int=None):
        '''Returns whether or not a hit is "valid", i.e. the starts and stops match, or the amount 
        of overlap with the GBFF file entry is greater than 50 percent.'''
        if (hit.strand != strand):
            return False
        if (hit.start == start) or (hit.stop == stop):
            return True
        if hit.percent_overlap > 50:
            return True
        return False

    def get_top_hit(self, query):
        '''
        '''
        df = self.gbff_file.to_df(contig_number=int(query.contig_number))
        start, stop, strand = int(query.start), int(query.stop), int(query.strand)
        if len(df) == 0: # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
            return {'overlap':None}, None

        overlap = lambda row : len(np.intersect1d(np.arange(start, stop), np.arange(row.start, row.stop)))
        df['overlap'] = df.apply(overlap, axis=1)
        df['percent_overlap'] = 100 * (df.overlap / (stop - start))
        df = df.sort_values('overlap', ascending=False)
        # Generate a mask to filter the valid hits. 
        mask = df.apply(lambda hit: Organism.is_valid_hit(hit, start=start, stop=stop, strand=strand), axis=1)

        hit = dict()
        hit['n_hits_same_strand'] = (df[df.strand == strand].overlap > 0).sum().item()
        hit['n_hits_opposite_strand'] = (df[df.strand != strand].overlap > 0).sum().item()
        hit['n_valid_hits'] = mask.sum().item()
        hit['n_hits'] = (df.overlap > 0).sum().item()

        if hit['n_valid_hits'] > 0:
            hit['overlap'] = None
            hit.update(df[mask].to_dict(orient='records')[0])

        search_results_df = df[mask] if (len(df[mask]) > 0) else None
        return hit, search_results_df # Return the top hit, as well as all other valid hits.


    def search_gbff_file(self, df:pd.DataFrame, **kwargs):
        '''Look for sequences in the input GBFF file which overlap somewhere in the specified range.'''  
        
        info_df, mask = list(), list()
        for query in tqdm(list(df.itertuples()), desc='search_gbff_file'):
            hit, self.gbff_search_results[query.Index] = self.get_top_hit(query)
            hit['id'] = query.Index
            info_df.append(hit)
        info_df = pd.DataFrame(info_df).set_index('id')
        mask = ~info_df.overlap.isnull() # This will be null if there are no hits. 
        return info_df[mask], df[~mask]

    def get_gbff_search_result(self, id_:str) -> pd.DataFrame:
        return self.gbff_search_results.get(id_, None)

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
        info_df = blast_df.copy()

        return info_df[mask], df[~mask]


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

        info_df = blast_df.copy()
        info_df['left_aligned'] = (info_df.query_alignment_start == 1) & (info_df.subject_alignment_start == 1)
        info_df['right_aligned'] = (info_df.query_alignment_end == info_df.query_sequence_length - 1) & (info_df.subject_alignment_end == info_df.subject_sequence_length)

        return info_df[mask], df[~mask]


    def label(self):
        rna_features = [feature for feature in GBFFFile.features if 'RNA' in feature]

        label_info = dict()
        label_info['match'], proteins_df = self.find_matches(self.proteins_df)
        label_info['error'], proteins_df = self.find_errors(proteins_df)

        gbff_search_info_df, label_info['inter'] = self.search_gbff_file(proteins_df)
        label_info['pseudo'] = gbff_search_info_df[gbff_search_info_df.pseudo].dropna(axis=1, how='all')
        label_info['rna'] = gbff_search_info_df[gbff_search_info_df.feature.isin(rna_features)].dropna(axis=1, how='all')

        labels = dict()
        for label, info_df in label_info.items():
            labels.update({id_:label for id_ in info_df.index})
            print(f'Organism.label: Found {len(info_df)} sequences in the input genome with the "{label}" label.')
        print(f'Organism.label: Successfully labeled {len(labels)} of the {len(self.proteins_df)} proteins.')

        unlabeled_ids = [id_ for id_ in proteins_df.index if (id_ not in labels)]
        labels.update({id_:'none' for id_ in unlabeled_ids}) # Account for the remaining sequences.
        label_info['none'] = gbff_search_info_df[gbff_search_info_df.index.isin(unlabeled_ids)]

        self.labels = labels
        self.label_info = label_info


    # def find_intergenic(self, df:pd.DataFrame, allowed_overlap:int=30):
    #     '''A GTDB ORF is intergenic if it either (1) does not overlap with any other nCDS element in the NCBI reference or (2) the overlap with a 
    #     protein (non-ppseudo) in the reference genome is no greater than the specified margin. 
    #     https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-15-721
    #     https://pmc.ncbi.nlm.nih.gov/articles/PMC525685/
    #     '''

    #     def is_intergenic(hit:dict, start:int=None, stop:int=None) -> (bool, dict):
    #         intergenic = True
    #         if hit['overlap'] is None:
    #             intergenic = True
    #         # elif hit['pseudo']:
    #         #     intergenic = True
    #         elif hit['percent_overlap'] > 50:
    #             intergenic = False
    #         elif hit['overlap'] > allowed_overlap:
    #             intergenic = False
    #         hit['intergenic'] = intergenic
    #         # hit['overlap'] = 0 # Make sure to mark the hit as invalid by setting the overlap to None. 
    #         return intergenic, hit

    #     # Thi
    #     mask, info_df = self.search_gbff_file(df, is_intergenic, psuedo=None)  
    #     self.label_info['inter'] = info_df
    #     return df[mask], df[~mask]