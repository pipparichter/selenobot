''''''
import os
import pandas as pd
import numpy as np
import copy
from typing import NoReturn, List, Tuple, Dict
import re
from selenobot.utils import dataframe_from_fasta
from Bio.Seq import Seq
import copy
from tqdm import tqdm
import scipy.stats
from typing import Tuple, List
import time

START_CODONS = {'+':['ATG', 'GTG', 'TTG'], '-':['CAT', 'CAC', 'CAA']}
STOP_CODONS = {'+':['TAA', 'TAG', 'TGA'], '-':['TTA', 'CTA', 'TCA']}

def get_scaffold_id(metadata_df:pd.DataFrame, id_:str=None) -> str:
    return metadata_df[metadata_df['id'] == id_]['scaffold_id'].item()

def get_contig(genome_df:pd.DataFrame, scaffold_id:str=None) -> str:
    # scaffold_id = re.escape(scaffold_id)
    genome_df['id'] = genome_df['id'].astype(type(scaffold_id))
    return genome_df[genome_df['id'] == scaffold_id]['seq'].item()

# TODO: Write now these classes are written in a way which requires a genome to be loaded, but does not explicitly raise
# exceptions if one is not loaded. I should either make the reliance on a genome explicit, or make it so that things work without it. 


class Sequence():
    attrs = ['start', 'stop', 'strand', 'scaffold_id', 'desc']

    def __init__(self, id_:str, metadata_df:pd.DataFrame, genome_df:pd.DataFrame=None):
        self.id_ = id_

        row = metadata_df[metadata_df['id'] == self.id_].to_dict(orient='records')[0]

        for attr in Sequence.attrs:
            setattr(self, attr, row.get(attr, None))
        # Make sure the type is correct for the start and stop positions. 
        self.start = int(self.start) - 1 # Start is one-indexed, so convert to zero-indexed.
        self.stop = int(self.stop)
        # Store the start and stop values a second time, in case the sequence is extended. 
        self._start, self._stop = self.start, self.stop

        self.start_codons = START_CODONS[self.strand]
        self.stop_codons = STOP_CODONS[self.strand]

        # If a genome is provided, load the contig.
        contig = get_contig(genome_df, scaffold_id=self.scaffold_id) if (genome_df is not None) else None
        self.contig = contig

    def __str__(self):
        # return self.id_ if self.desc is None else f'{self.id_}: {self.desc}'
        return self.seq() 

    def __eq__(self, seq):
        '''Define equality between two sequence objects to mean (1) the sequences match and (2) the IDs match.'''
        return (self.id_ == seq.id_) and (self.seq() == id_.seq())

    def __hash__(self):
        # https://stackoverflow.com/questions/9089400/set-in-operator-uses-equality-or-identity
        return hash((self.id_, self.seq())) # Is this going to be weird if self.seq() is None?

    def __len__(self):
        return self.stop - self.start

    def seq(self):
        '''Retrieve the gene's nucleotide sequence from the loaded contig.'''
        # TODO: Should this raise an error if no DNA is loaded? Or perhaps just return an empty string?
        # return self.contig[self.start:self.stop] if (contig is not None) else ''
        return self.contig[self.start:self.stop] 

    def start_codon(self):
        return self.seq()[:3] if self.strand == '+' else self.seq()[-3:]

    def stop_codon(self):
        return self.seq()[:3] if self.strand == '-' else self.seq()[-3:]

    def has_valid_stop_codon(self):
        return self.stop_codon() in self.stop_codons

    def has_valid_start_codon(self):
        return self.start_codon() in self.start_codons
    
    def strand(self, opposite:bool=False):
        return self.strand if (not opposite) else ('+' if self.strand == '-' else '-')
    
    def opposite_strand(self):
        return 
    
    def downstream(self):
        return self.contig[:self.start] if self.strand == '-' else self.contig[self.stop:]

    def reverse_complement(self) -> str:
        '''Converts the input DNA strand to its reverse complement.'''
        seq = Seq(self.seq())
        seq = seq.reverse_complement()
        return str(seq)

    def translate(self):
        '''Translate a nucleotide sequence.'''
        seq = self.reverse_complement() if self.strand == '-' else self.seq()
        # Translate the sequence using the BioPython module. 
        seq = Seq(seq).translate(to_stop=False) # Can't set cds=True because it throws an error with in-frame stop codons.  
        
        assert seq[-1] == '*', 'extend.translate: The last symbol in the amino acid sequence should be *, indicating a translational stop.'
        
        seq = seq[:-1] # Remove the terminal * character. 
        seq = str(seq).replace('*', 'U') # Replace the in-frame stops with selenocysteine.
        seq = 'M' + seq[1:] # Not sure why this is always methionine in NCBI, but doesn't always get translated as M. 

        return seq

    def info(self):
        info['id'] = self.id_ 
        info['scaffold_id'] = self.scaffold_id
        info['strand'] = self.strand
        info['stop'] = self.stop 
        info['start'] = self.start 
        info['contig'] = self.contig
        return info


class ContextSequence(Sequence):

    # def __init__(self, id_:str, start:int=None, stop:int=None, contig:str=None, strand:str=None):
    def __init__(self, id_:str, metadata_df:pd.DataFrame, genome_df:pd.DataFrame=None):
        
        super().__init__(id_, metadata_df, genome_df=genome_df)
        self.extended = False

        # Extract only the metadata for genes on the same scaffold (i.e. contig)
        metadata_df = metadata_df[metadata_df.scaffold_id == self.scaffold_id]

        # Note that the neighbors don't account for sequences which are already overlapping
        self.neighbors = {'-':dict(), '+':dict()}
        for strand in ['+', '-']:
            # Filter the DataFrame to get only those genes on the same contig and specified strand. 
            strand_df = metadata_df[metadata_df.strand == strand] 
            left_ids = strand_df[strand_df.stop < self.start].sort_values('stop')['id'].values
            right_ids = strand_df[strand_df.start > self.stop].sort_values('start')['id'].values
            self.neighbors[strand]['right'] = None if len(right_ids) < 1 else Sequence(right_ids[0], metadata_df, genome_df=genome_df)
            self.neighbors[strand]['left'] = None if len(left_ids) < 1 else Sequence(left_ids[-1], metadata_df, genome_df=genome_df)

        # Get any sequences which overlap the current ContextSequence on either strand. This would be if the start or stop
        # location of another gene lies within the start or stop of the current gene. 
        self.overlaps = {'-':[], '+':[]} 
        overlap_df = []
        overlap_df.append(strand_df[(strand_df.start < self.stop) & (strand_df.start > self.start)])
        overlap_df.append(strand_df[(strand_df.stop < self.stop) & (strand_df.stop > self.start)])
        overlap_df = pd.concat(overlap_df)
        for row in overlap_df.to_dict(orient='records'):
            self.overlaps[row['strand']].append(Sequence(row['id'], metadata_df, genome_df=genome_df))

    def next(self, opposite:bool=False):
        # Get downstream gene on the relevant strand (either the same strand as the main ContextSequence, or the opposite strand).
        return neighbors[self.strand(opposite=opposite)]['left'] if self.strand == '-' else neighbors[self.strand(opposite=opposite)]['right'] 

    def overlap(self, seq:Sequence) -> int:
        '''Determine if the sequence overlaps with the input ContextSequence.'''
        # Overlap is present if the neighbor stop is to the right of this gene's start, so this value would be negative. 
        overlap = min(0, self.start - seq.stop)
        # Overlap is present if the neighbor start is to the left of theis gene's stop, so this value would be negative. 
        overlap = min(overlap, seq.start - self.stop)
        return abs(overlap)

    def intergenic_distance(self, seq:Sequence):
        # TODO: Might be a good idea to make this signed, i.e. make it so that intergenic distance to the left is negative. 
        if (seq.stop < self.start) and (seq.start < self.start): # Then the query gene is to the left of self.
            return self.start - seq.stop
        elif (seq.start > self.stop) and (seq.stop > self.stop): # Then the query gene is to the right of self.
            return seq.start - self.stop
        else: # TODO: Make sure this handles overlaps.
            None

    def downstream(self):
        # Get the codons from the stop location (the end of the stop codon) to the end of the scaffold. 
        return self.contig[:self.start] if self.strand == '-' else self.contig[self.stop:]

    def extendable(self):
        # Check the case in which the gene is at the end of the contig.
        if len(self.downstream()) < 3: # Make sure there actually *is* a downstream region.
            return False
        return True

    def extend(self, verbose:bool=False):

        assert self.extendable(), f'ContextSequence.extend: No extension possible for ContextSequence {self.id_}.'

        downstream = self.downstream()
        # Need to account for the case in which the gene is at the end of the contig.

        iter_start = len(downstream) - 3 if (self.strand == '-') else 0
        iter_stop = len(downstream) if (self.strand == '-') else 3
        step = -3 if (self.strand == '-') else 3

        ext_length = 0
        for i in range(len(downstream) // 3):
            codon = downstream[iter_start + (i * step):iter_stop + (i * step)]
            ext_length += 3
            if codon in self.stop_codons:
                break
        
        cseq_extended = copy.deepcopy(self)
        if cseq_extended.strand == '-':
            cseq_extended.start = cseq_extended.start - ext_length
        elif cseq_extended.strand == '+':
            cseq_extended.stop = cseq_extended.stop + ext_length

        cseq_extended.extended = True
        # Add new overlaps if the extended sequence now interferes with either downstream gene. 
        next_seq, next_seq_opposite = self.next(), self.next(opposite=True)
        strand, strand_opposite = self.strand(), self.strand(opposite=True)

        if (next_seq not in self.overlaps[strand]) and (cseq_extended.overlap(next_seq) > 0):
            cseq_extended.overlaps[strand].append(next_seq)
        if (next_seq_opposite not in self.overlaps[strand_opposite]) and (cseq_extended.overlap(next_seq_opposite) > 0):
            cseq_extended.overlaps[strand_opposite].append(next_seq_opposite)

        return cseq_extended

    def extension(self):
        '''Return only the extended region of an extended ContextSequence, i.e. the region downstream of the original
        stop codon.'''
        # Check to make sure the extend method has been called on the sequence.  
        if not self.extended:
            return None
        cseq_extension = copy.deepcopy(self) # ContextSequence to store the extension region only.
        # The pre-extension stop and start locations are stored in the start_ and stop_ attributes. 
        # Depending on the direction of the extension (i.e. the strand), adjust the stop and start attributes.
        if self.strand == '+':
            cseq_extension.start = self._stop
        elif self.strand == '-':
            cseq_extension.stop = self._start
        
        return cseq_extension

    def truncate(self):

        trunc_cseq = copy.deepcopy(self)
        sec_idx = trunc_cseq.translate().index('U')
        if self.strand == '+':
            trunc_cseq.stop = trunc_cseq.start + ((sec_idx + 1) * 3) # This is the new end of the truncated sequence. 
        elif self.strand == '-':
            trunc_cseq.start = trunc_cseq.stop - ((sec_idx + 1) * 3)

        return trunc_cseq

    # def split(self):
    #     '''Split the ContextSequence in two, which is useful for the cofitness positive control.'''
    #     left_cseq, right_cseq = copy.deepcopy(self), copy.deepcopy(self)
    #     split_loc = (self.stop - self.start) // 2
        
    #     left_cseq.stop = self.start + split_loc
    #     left_cseq.stop_ = self.start + split_loc
    #     right_cseq.start = self.start + split_loc
    #     right_cseq.start_ = self.start + split_loc

    #     return left_cseq, right_cseq

    def info(self):
        info = Sequence.info(self)
        info['extended'] = self.extended
        return info



class FitnessBrowserSequence(ContextSequence):
    '''Object for managing Fitness Browser data, including experimental fitness values and t-values.'''
    # The columns in the fitness data which do not contain actual fitness values. 
    non_experiment_cols = ['id', 'scaffold_id', 'pos', 'strand']
    
    def __init__(self, id_:str, metadata_df:pd.DataFrame, fitness_df:pd.DataFrame, genome_df:pd.DataFrame=None, t_values_df:pd.DataFrame=None, load_neighbors:bool=True, low_memory:bool=True):

        super().__init__(id_, metadata_df, genome_df=genome_df, load_neighbors=load_neighbors)

        # Avoid copying the whole fitness_df if in low memory mode.
        self._fitness_df = None if low_memory else copy.deepcopy(fitness_df)

        self.load_fitness(fitness_df)

        if t_values_df is None:
            self.t_values = None
        else:
            self.load_t_values(t_values_df)
    
    @classmethod
    def from_cseq(cls, cseq:ContextSequence, fitness_df:pd.DataFrame, t_values_df:pd.DataFrame=None, low_memory:bool=True):
        
        fbseq = FitnessBrowserSequence.__new__(cls)
        for attr, value in vars(cseq).items():
            setattr(fbseq, attr, copy.deepcopy(value))

        # Avoid copying the whole fitness_df if in low memory mode.
        fbseq._fitness_df = None if low_memory else copy.deepcopy(fitness_df)

        fbseq.load_fitness(fitness_df)

        if t_values_df is None:
            fbseq.t_values = None
        else:
            fbseq.load_t_values(t_values_df)

        return fbseq
        

    def hits(self) -> int:
        '''Return the number of fitness hits for the sequence.'''
        return len(self.fitness)
    
    def overlapping_hits(self) -> int:
        # Get the locus IDs for all hits in the fitness data. 
        hit_ids = self.fitness.index
        # if np.issubdtype(hit_ids.dtype, np.number):
        hit_ids = hit_ids[~pd.isnull(self.fitness.index)]
        hit_ids = hit_ids.values[hit_ids.values != self.id_]
        return len(hit_ids)

    def mean(self, experiments:List[str]=None):
        fitness = copy.deepcopy(self.fitness)
        fitness = fitness.mean(axis=0)
        if experiments is not None:
            fitness = fitness[experiments]
        return fitness.values.ravel()

    def err(self, experiments:List[str]=None):
        fitness = copy.deepcopy(self.fitness)
        fitness = fitness.std(axis=0) / np.sqrt(self.hits())
        if experiments is not None:
            fitness = fitness[experiments]
        return fitness.values

    def experiments(self, significant_phenotypes:bool=False, t_value_threshold:float=4, fitness_threshold:float=0.5) -> List[str]:
        
        experiments = self.fitness.columns.values 
        
        if significant_phenotypes:
            assert self.t_values is not None, 'FitnessBrowserSequence.experiments: t-values must be loaded in order to determine significance.'
            significant = np.abs(self.mean()) > fitness_threshold
            significant = np.logical_and(significant, np.abs(self.t_values.values.ravel() > t_value_threshold))
            experiments = experiments[significant]

        return experiments

    def significance(self, t_value_threshold:float=4, fitness_threshold:float=0.5):
        '''Computes a "significance index" for a particular locus, which I am defining as the ratio
        of experiments with significant phenotyes to all performed experiments.'''
        total_experiments = len(self.experiments())
        significant_experiments = len(self.experiments(significant_phenotypes=True, t_value_threshold=t_value_threshold, fitness_threshold=fitness_threshold))
        return significant_experiments / total_experiments

    def load_t_values(self, t_values_df:pd.DataFrame):
        # Load in the t-values for the specific gene.
        t_values_df = t_values_df[t_values_df['id'] == self.id_]
        # TODO: Look into why a locus might not have a t-value. 
        # Not all loci seem to have computed t-values, so make sure to handle that case. 
        self.t_values = None if (len(t_values_df) == 0) else t_values_df[self.experiments()]

    def load_fitness(self, fitness_df):
        # Extract all hits from the fitness data. DON'T use the locus ID alone, as this will not work with the extended sequences. 
        fitness_df = fitness_df[fitness_df.scaffold_id == self.scaffold_id]
        fitness_df = fitness_df[fitness_df.strand == self.strand]
        fitness_df = fitness_df[fitness_df['pos'] <= self.stop]
        fitness_df = fitness_df[fitness_df['pos'] >= self.start]

        # if 'id' in fitness_df.columns:
        fitness_df = fitness_df.set_index('id') # Make sure to keep the ID in the DataFrame.
        
        self.fitness = fitness_df[[c for c in fitness_df.columns if c not in FitnessBrowserSequence.non_experiment_cols]]
        self.positions = fitness_df[['pos']]

    def extend(self):

        assert self._fitness_df is not None, 'FitnessBrowserSequence.extend: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
        ext_fbseq = ContextSequence.extend(self) # Extend the sequence. copy.deepcopy is already called here.  
        ext_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
        return ext_fbseq

    def extension(self):
        '''Return only the extended region of an extended FitnessBrowserSequence, i.e. the region downstream of the original
        stop codon.''' 
        assert self._fitness_df is not None, 'FitnessBrowserSequence.extension: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
        ext_only_fbseq = ContextSequence.extension(self) # Get the extended region. copy.deepcopy is already called here. 
        ext_only_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
        return ext_only_fbseq

    def split(self):
        assert self._fitness_df is not None, 'FitnessBrowserSequence.extension: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
        left_fbseq, right_fbseq = ContextSequence.split(self) # Get the split sequences. copy.deepcopy is already called here. 
        left_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
        right_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
        return left_fbseq, right_fbseq

    def cofitness(self, fbseq): # -> Tuple[float, scipy.stats.LinregressResult]:
        '''Compute the cofitness score between the two FitnessBrowserSequence objects. As defined in the Price et. al. 2018 paper,
        the cofitness score is the absolute value of the r-value from regression.'''
        linreg = scipy.stats.linregress(self.mean(), fbseq.mean())
        return float(abs(linreg.rvalue)), linreg



def save_seqs(seqs:List[ContextSequence], path:str):
    '''Save information from a list of context sequences to a CSV file.'''
    rows = [seq.info() for seq in seqs]  
    df = pd.DataFrame(rows)  
    df = df.set_index('id')
    df.to_csv(path)


def load_cseqs(metadata_df:pd.DataFrame, genome_df:pd.DataFrame=None, use_ids:List[str]=None) -> List[ContextSequence]:
    '''Load all sequences in a gene metadata DataFrame as ContextSequence objects.'''

    ids = metadata_df['id'].unique() # Extract all locus IDs from the metadata DataFrame.
    if (use_ids is not None): # Skip the ID if a specific list of IDs is specified, and the ID is not in a list.
        ids = [id_ for id_ in ids if id_ in use_ids]
 
    cseqs = []
    for id_ in tqdm(ids, desc='load_cseqs'):
        if (use_ids is not None) and (id_ not in use_ids):
            continue # Skip the ID if a specific list of IDs is specified, and the ID is not in a list.
        cseq = ContextSequence(id_, metadata_df, genome_df=genome_df)
        cseqs.append(cseq) # Add the ContextSequence to the list. 
    return cseqs


def load_fbseqs(metadata_df:pd.DataFrame, fitness_df:pd.DataFrame, t_values_df:pd.DataFrame=None, genome_df:pd.DataFrame=None, use_ids:List[str]=None, low_memory:bool=False):
    '''Load all sequences in a gene metadata DataFrame, as well as their corresponding fitness data, as 
    FitnessBrowserSeq objects.'''
    ids = metadata_df['id'].unique() # Extract all locus IDs from the metadata DataFrame. 
    if (use_ids is not None): # Skip the ID if a specific list of IDs is specified, and the ID is not in a list.
        ids = [id_ for id_ in ids if id_ in use_ids]

    fbseqs = []
    for id_ in tqdm(ids, desc='load_fbseqs'):
        fbseq = FitnessBrowserSequence(id_, metadata_df, fitness_df, t_values_df=t_values_df, genome_df=genome_df, low_memory=low_memory)
        fbseqs.append(fbseq) # Add the ContextSequence to the list. 
    return fbseqs

