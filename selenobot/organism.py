import os 
import pandas as pd 
import numpy as np
import subprocess
import shutil
import math
from tqdm import tqdm
from Bio.Seq import Seq

from selenobot.tools import BLAST
from selenobot.files import BLASTFile, FASTAFile, GBFFFile, fasta_file_parser_gtdb
from selenobot.utils import apply_gtdb_dtypes

class Organism():

    def __init__(self, genome_id:str, species:str, dir_:str='../data/model_organisms/'):
        '''Initialize an Organism object.'''

        self.code_name = (species.split()[0][0] + species.split()[-1][:3]).lower() 
        self.dir_ = dir_
        self.genome_id = genome_id
        self.species = species

        self.path = os.path.join(dir_, f'gtdb_{self.code_name}_protein.faa')
        self.ref_path = os.path.join(dir_, f'ncbi_{self.code_name}_genomic.gbff')
        
        ref_file = GBFFFile(self.ref_path) 

        self.df = apply_gtdb_dtypes(FASTAFile(self.path).to_df(parser=fasta_file_parser_gtdb))
        self.df['contig_id'] = [id_.split('.')[0] for id_ in self.df.index]

        self.ref_df = apply_gtdb_dtypes(ref_file.to_df()) # Will work to fix the data types in this DataFrame too. 
        self.contigs = ref_file.contigs

        self.labels = dict()
        self.label_info = dict()

        self.search_results_df = None

        self.df = self.add_start_stop_codons(self.df)
        self.ref_df = self.add_start_stop_codons(self.ref_df)

    def __repr__(self):
        return f'{self.species.split()[0][0]}. {self.species.split()[-1]}'

    def __str__(self):
        return self.code_name

    def to_df(self, max_seq_length:int=None, label:str=None):
        df = self.df.copy() 
        df['code_name'] = self.code_name
        df['genome_id'] = self.genome_id 
        df['species'] = self.species

        if len(self.labels) > 0:
            assert len(self.labels) == len(df), f'Organism: There are {len(self.labels)} labels and {len(df)} entries in the protein DataFrame.'
            df = df.merge(pd.DataFrame({'label':self.labels}), right_index=True, left_index=True, how='left')

        if (max_seq_length is not None):
            df = df[df.seq.apply(len) < max_seq_length]

        if (self.search_results_df is not None):
            df = df.merge(self.search_results_df, left_index=True, right_index=True)
            ref_df = self.ref_df.copy().rename(columns={col:'ref_' + col for col in self.ref_df.columns})
            df = df.merge(ref_df, left_on='locus_tag', right_index=True, how='left')
            df = df.dropna(how='all')

        return df

    def get_hit(self, query):
        '''
        '''
        start, stop, strand = int(query.start), int(query.stop), int(query.strand)
        
        df = self.ref_df.copy() 
        df = df[df.contig_id == query.contig_id]
        # Filter out everything which definitely has no overlap.
        df = df[~(df.start > stop) & ~(df.stop < start)]

        if len(df) == 0: # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
            return {'n_hits':0, 'n_valid_hits':0, 'feature':None, 'locus_tag':None, 'pseudo':None}

        df['valid_hit'] = ((df.stop == stop) | (df.start == start)) & (df.strand == strand)
        n_hits, n_valid_hits = len(df), df.valid_hit.sum().item() 

        if n_valid_hits > 1:
            df = df[df.valid_hit]
            df['length_diff'] = np.abs((df.stop - df.start) - (stop - start))
            df = df.sort_values(by='length_diff').iloc[[0]]
        # assert n_valid_hits < 2, f'Organism.get_top_hit: Expected no more than one valid hit, found {n_valid_hits}.'

        if n_valid_hits == 0:
            return {'n_hits':n_hits, 'n_valid_hits':0, 'feature':None, 'locus_tag':None, 'pseudo':None} 

        feature = df['feature'].iloc[0]
        pseudo = df['pseudo'].iloc[0]
        locus_tag = df.index[0]
        return {'n_hits':n_hits, 'n_valid_hits':n_valid_hits, 'feature':feature, 'locus_tag':locus_tag, 'pseudo':pseudo}  


    def search(self, df:pd.DataFrame, **kwargs):
        '''Look for sequences in the input GBFF file which overlap somewhere in the specified range.'''  
        
        hits_df, mask = list(), list()
        for query in tqdm(list(df.itertuples()), desc='search'):
            hit = self.get_hit(query) # Get the hit with the biggest overlap, with a preference for "valid" hits. 
            hit['id'] = query.Index
            hits_df.append(hit)
        return pd.DataFrame(hits_df).set_index('id')


    def label(self):
        rna_features = [feature for feature in GBFFFile.features if 'RNA' in feature]
        misc_features = ['misc_feature', 'mobile_element', 'repeat_region']

        label_info = dict()
        hits_df = self.search(self.df)
        
        label_info['inter'] = hits_df[hits_df.n_hits == 0]
        label_info['error'] = hits_df[(hits_df.n_hits > 0) & (hits_df.n_valid_hits == 0)]
        label_info['pseudo'] = hits_df[hits_df.pseudo == True]
        label_info['cds'] = hits_df[(hits_df.feature == 'CDS') & ~(hits_df.pseudo == True)]
        label_info['rna'] = hits_df[hits_df.feature.isin(rna_features)]
        label_info['misc'] = hits_df[hits_df.feature.isin(misc_features)]

        labels = dict()
        for label, info_df in label_info.items():
            labels.update({id_:label for id_ in info_df.index})
            print(f'Organism.label: Found {len(info_df)} sequences in the input genome with the "{label}" label.')

        self.labels = labels
        self.search_results_df = hits_df


    def get_nt_seq(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, error:str='ignore'):
        nt_seq = self.contigs[contig_id] 
        # Pretty sure the stop position is non-inclusive, so need to shift it over.
        nt_seq = nt_seq[start - 1:stop] 
        nt_seq = str(Seq(nt_seq).reverse_complement()) if (strand == -1) else nt_seq # If on the opposite strand, get the reverse complement. 

        if( len(nt_seq) % 3 == 0) and (error == 'raise'):
            raise Exception(f'GBFFFile.get_nt_seq: Expected the length of the nucleotide sequence to be divisible by three, but sequence is of length {len(nt_seq)}.')

        return nt_seq

    def get_stop_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
        return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[-3:]
    
    def get_start_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
        return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[:3]

    def add_start_stop_codons(self, df:pd.DataFrame) -> pd.DataFrame:
        '''Add start and stop codons to the input DataFrame. Assumes the DataFrame contains, at least, columns for the 
        nucleotide start and stop positions, the strand, and the contig ID.'''
        start_codons, stop_codons = ['ATG', 'GTG', 'TTG'], ['TAA', 'TAG', 'TGA']

        df['stop_codon'] = [self.get_stop_codon(**row) for row in df.to_dict(orient='records')]
        df['start_codon'] = [self.get_start_codon(**row) for row in df.to_dict(orient='records')]

        df.stop_codon = df.stop_codon.apply(lambda c : 'none' if (c not in stop_codons) else c)
        df.start_codon = df.start_codon.apply(lambda c : 'none' if (c not in start_codons) else c)

        return df


    # def download_ncbi_data(self):

    #     output_path = os.path.join(self.dir_, 'ncbi.zip')
    #     cmd = f'datasets download genome accession {self.genome_id} --filename {output_path} --include protein,gbff,genome --no-progressbar'
    #     subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    #     subprocess.run(f'unzip -o {output_path} -d {self.dir_}', shell=True, check=True, stdout=subprocess.DEVNULL)
            
    #     file_names = ['protein.faa', 'genomic.gbff', '*genomic.fna']
    #     src_paths = [os.path.join(self.dir_, 'ncbi_dataset/data', self.genome_id, file_name) for file_name in file_names]
    #     dst_paths = [self.ncbi_proteins_path, self.ncbi_gbff_path, self.genome_path]
    #     for src_path, dst_path in zip(src_paths, dst_paths):
    #         subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True)

    #     shutil.rmtree(os.path.join(self.dir_, 'ncbi_dataset'))
    #     os.remove(output_path)
    #     return GBFFFile(self.ref_path)