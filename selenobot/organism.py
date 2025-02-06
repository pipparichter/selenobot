import os 
import pandas as pd 
import numpy as np
import subprocess
import shutil
import math
from tqdm import tqdm
from Bio.Seq import Seq
import pickle 

from selenobot.tools import BLAST
from selenobot.files import BLASTFile, FASTAFile, GBFFFile, fasta_file_parser_prodigal
from selenobot.utils import apply_gtdb_dtypes


def get_organisms(species:list, from_gtdb:bool=False, overwrite:bool=False):

    path = '../data/model_organisms/organisms.pkl' if (not from_gtdb) else '../data/model_organisms/organisms_gtdb.pkl'
    dir_ = '../data/model_organisms/proteins' if (not from_gtdb) else '../data/model_organisms/proteins_gtdb'
    ref_dir_ = '../data/model_organisms/ref' 

    if (not os.path.exists(path)) or overwrite:
        organisms = list()
        for species_ in species:
            print(f'get_organisms: Creating Organism object for {species_}...')
            organism = Organism(species_, dir_=dir_, ref_dir=ref_dir_)
            organism.label()
            organisms.append(organism)
        with open(path, 'wb') as f:
            pickle.dump(organisms, f)
        print(f'get_organisms: Organism objects saved to {path}.')
    else:
        with open(path, 'rb') as f:
            organisms = pickle.load(f)
    return organisms


def get_code_name(species:str) -> str:
    return (species.split()[0][0] + species.split()[-1][:3]).lower() 


def download_ncbi_data(genome_metadata_df:pd.DataFrame, dir_:str='../data/model_organisms'):
    '''Dowload genomes and GBFF files for the organisms contained in the input DataFrame from NCBI.'''
    genomes_dir = os.path.join(dir_, 'genomes')
    proteins_ref_dir = os.path.join(dir_, 'proteins_ref')

    output_path = 'ncbi.zip'
    for row in genome_metadata_df.itertuples():
        print(f'download_ncbi_data: Downloading data for {row.species}.')

        code_name = get_code_name(row.species)
        protein_ref_path = os.path.join(proteins_ref_dir, f'{code_name}_genomic.gbff')
        genome_path = os.path.join(genomes_dir, f'{code_name}_genomic.fna')
        
        cmd = f'datasets download genome accession {row.Index} --filename {output_path} --include gbff,genome --no-progressbar'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(f'unzip -o {output_path} -d .', shell=True, check=True, stdout=subprocess.DEVNULL)
            
        file_names = ['genomic.gbff', '*genomic.fna']
        src_paths = [os.path.join('ncbi_dataset/data', row.Index, file_name) for file_name in file_names]
        dst_paths = [protein_ref_path, genome_path]

        for src_path, dst_path in zip(src_paths, dst_paths):
            subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True)
    # Clean up extra files. 
    shutil.rmtree(os.path.join('ncbi_dataset'))
    os.remove('README.md')
    os.remove('md5sum.txt')
    os.remove(output_path)



class Organism():

    def __init__(self, species:str, dir_:str='../data/model_organisms/proteins/', ref_dir:str='../data/model_organisms/ref/'):
        '''Initialize an Organism object.'''

        self.code_name = get_code_name(species)
        self.dir_ = dir_ 
        self.species = species

        self.path = os.path.join(dir_, f'{self.code_name}_protein.faa')
        self.ref_path = os.path.join(ref_dir, f'{self.code_name}_genomic.gbff')
        
        ref_file = GBFFFile(self.ref_path) 

        self.df = apply_gtdb_dtypes(FASTAFile(self.path).to_df(parser=fasta_file_parser_prodigal))
        self.df['contig_id'] = [id_.split('.')[0] for id_ in self.df.index]

        self.ref_df = apply_gtdb_dtypes(ref_file.to_df()) # Will work to fix the data types in this DataFrame too. 
        self.contigs = ref_file.contigs

        self.labels = dict()
        self.label_info = dict()

        self.search_results_df = None

        self.df = self.add_start_stop_codons(self.df)
        self.ref_df = self.add_start_stop_codons(self.ref_df)
        self.df = Organism.add_length(self.df)
        self.ref_df = Organism.add_length(self.ref_df)

    def __repr__(self):
        return f'{self.species.split()[0][0]}. {self.species.split()[-1]}'

    def __str__(self):
        return self.code_name
    
    def __len__(self):
        return len(self.df)

    def to_df(self):
        df = self.df.copy() 
        df['code_name'] = self.code_name
        df['species'] = self.species

        if len(self.labels) > 0:
            df = df.merge(pd.DataFrame({'label':self.labels}), right_index=True, left_index=True, how='left')
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

    @staticmethod
    def add_length(df:pd.DataFrame):
        '''Add sequence length (in amino acids) to the DataFrame. Can't just use seq.apply(len) because forcing
        the sequences to be strings causes null sequences (e.g., in the case of non-CDS features) to be 'nan'.'''
        # This also gets lengths for pseudogenes. 
        lengths = list()
        for row in df.itertuples():
            # The reference DataFrame will have a feature column, but the Prodigal-produced one will not. 
            # Only try to get sequence lengths for the CDS features from the NCBI reference. 
            feature = getattr(row, 'feature', None)
            if (feature == 'CDS') or (feature is None):
                lengths.append((row.stop - row.start) // 3) # The start and stop indices are in terms of nucleotides. 
            else:
                lengths.append(None)
        df['length'] = lengths 
        return df

