import os 
import pandas as pd 
import numpy as np
import subprocess
import shutil
import math
from tqdm import tqdm

from selenobot.tools import BLAST
from selenobot.files import BLASTFile, FASTAFile, GBFFFile, fasta_file_parser_gtdb
from selenobot.utils import apply_gtdb_dtypes

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

        self.proteins_df = FASTAFile(self.gtdb_proteins_path).to_df(parser=fasta_file_parser_gtdb)
        self.proteins_df.seq = self.proteins_df.seq.str.replace(r'*', '') # Remove the terminal * character.
        self.protein_df = apply_gtdb_dtypes(self.proteins_df)

        self.labels = dict()
        self.label_info = dict()

        if os.path.exists(self.ncbi_gbff_path):
            self.gbff_file = GBFFFile(self.ncbi_gbff_path)
        else:
            self.gbff_file = self.download_ncbi_data()

        self.gbff_search_results = dict()
    
    def size(self, source:str='gtdb', pseudo:bool=None):
        if source == 'gtdb':
            fasta_file = FASTAFile(self.gtdb_proteins_path) 
            return len(fasta_file)
        elif source == 'ncbi':
            gbff_df = self.gbff_file.to_df(pseudo=pseudo)
            return len(gbff_df)


    def __eq__(self, code_name:str):
        return self.code_name == code_name

    def to_df(self, max_seq_length:int=None, label:str=None):
        df = self.proteins_df 
        df['code_name'] = self.code_name
        df['genome_id'] = self.genome_id 
        df['species'] = self.species
        if len(self.labels) > 0:
            assert len(self.labels) == len(df), f'Organism: There are {len(self.labels)} labels and {len(df)} entries in the protein DataFrame.'
            df = df.merge(pd.DataFrame({'label':self.labels}), right_index=True, left_index=True, how='left')
        if (max_seq_length is not None):
            df = df[df.seq.apply(len) < max_seq_length]
        if (label is not None):
            df = df[df.label == label]
            label_info_df = self.label_info[label]
            label_info_df = label_info_df.rename(columns={'seq':'ref_seq'})
            label_info_df = label_info_df.drop(columns=[col for col in label_info_df.columns if (col in df.columns)])
            df = df.merge(label_info_df, right_index=True, left_index=True, how='left')
        df.strand = df.strand.apply(pd.to_numeric)
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

    def get_top_hit(self, query):
        '''
        '''
        start, stop, strand = int(query.start), int(query.stop), int(query.strand)
        
        df = self.gbff_file.to_df(contig_number=int(query.contig_number))
        if len(df) == 0: # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
            return {'overlap':0, 'n_hits':0, 'n_valid_hits':0}

        # Filter out everything which definitely has no overlap to speed up search.
        df = df[~(df.start > stop)]
        df = df[~(df.stop < start)] 
        if len(df) == 0: # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
            return {'overlap':0, 'n_hits':0, 'n_valid_hits':0}

        overlap = lambda row : len(np.intersect1d(np.arange(start, stop), np.arange(row.start, row.stop)))
        df['overlap'] = df.apply(overlap, axis=1)
        df['same_start'] = (df.start == start)
        df['same_stop'] = (df.stop == stop)
        df['length_diff'] = (stop - start) - (df.stop - df.start)
        df = df.sort_values('overlap', ascending=False)
        df = df[df.overlap > 0].copy() # Filter out all instances of no overlap. 
        df['valid_hit'] = (df.same_stop | df.same_stop) & (df.strand == strand)

        # Generate a mask to filter the valid hits, which share either a start or stop position. 
        hit = dict()
        hit['n_valid_hits'] = df.valid_hit.sum().item()
        hit['n_hits'] = len(df)
        
        if len(df) == 0: # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
            return {'overlap':0, 'n_hits':0, 'n_valid_hits':0}

        df = df[df.valid_hit] if (hit['n_valid_hits'] > 0) else df
        hit.update(df.to_dict(orient='records')[0])
        return hit  # Return the top hit, as well as all other valid hits.


    def search(self, df:pd.DataFrame, **kwargs):
        '''Look for sequences in the input GBFF file which overlap somewhere in the specified range.'''  
        
        hits_df, mask = list(), list()
        for query in tqdm(list(df.itertuples()), desc='search'):
            hit = self.get_top_hit(query) # Get the hit with the biggest overlap, with a preference for "valid" hits. 
            hit['id'] = query.Index
            hits_df.append(hit)
        return pd.DataFrame(hits_df).set_index('id')


    def label(self):
        rna_features = [feature for feature in GBFFFile.features if 'RNA' in feature]
        misc_features = ['misc_feature', 'mobile_element', 'repeat_region']

        label_info = dict()
        hits_df = self.search(self.proteins_df)
        
        label_info['inter'] = hits_df[(hits_df.n_valid_hits == 0) & (hits_df.overlap < 30)]
        label_info['error'] = hits_df[(hits_df.n_hits > 0) & (hits_df.n_valid_hits == 0)]

        hits_df = hits_df[hits_df.n_valid_hits > 0]
        label_info['pseudo'] = hits_df[hits_df.pseudo == True].dropna(axis=1, how='all')
        label_info['cds'] = hits_df[(hits_df.feature == 'CDS') & ~(hits_df.pseudo == True)].dropna(axis=1, how='all')
        label_info['rna'] = hits_df[hits_df.feature.isin(rna_features)].dropna(axis=1, how='all')
        label_info['misc'] = hits_df[hits_df.feature.isin(misc_features)].dropna(axis=1, how='all') 

        labels = dict()
        for label, info_df in label_info.items():
            labels.update({id_:label for id_ in info_df.index})
            print(f'Organism.label: Found {len(info_df)} sequences in the input genome with the "{label}" label.')

        self.labels = labels
        self.label_info = label_info

