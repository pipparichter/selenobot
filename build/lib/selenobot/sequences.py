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

# TODO: This currently doesn't handle plasmids. 

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
    attrs = ['start', 'stop', 'strand', 'scaffold_id', 'desc', 'description']

    def __init__(self, id_:str, metadata_df:pd.DataFrame, genome_df:pd.DataFrame=None):
        self.id_ = id_

        row = metadata_df[metadata_df['id'] == self.id_].to_dict(orient='records')[0]

        for attr in Sequence.attrs:
            setattr(self, attr, row.get(attr, None))
        # Store the opposite strand to make some other operations a bit easier. 
        self.opposite_strand = '-' if (self.strand == '+') else '+'
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

    def fasta(self, file_type='fn'):
        info = dict()
        info['id'] = self.id_ 
        # info['scaffold_id'] = self.scaffold_id
        info['strand'] = self.strand
        info['stop'] = self.stop 
        info['start'] = self.start 
        # info['contig'] = self.contig
        info['seq'] = self.seq() if file_type == 'fn' else self.translate()
        return info


class ContextSequence(Sequence):

    # def __init__(self, id_:str, start:int=None, stop:int=None, contig:str=None, strand:str=None):
    def __init__(self, id_:str, metadata_df:pd.DataFrame, genome_df:pd.DataFrame=None):
        
        super().__init__(id_, metadata_df, genome_df=genome_df)
        self.extended = False
        self.ext_length = 0

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
        self.overlaps = []
        overlap_df = []
        overlap_df.append(metadata_df[(metadata_df.start < self.stop) & (metadata_df.start > self.start)]) # Overlapping head.
        overlap_df.append(metadata_df[(metadata_df.stop < self.stop) & (metadata_df.stop > self.start)]) # Overlapping tail. 
        overlap_df = pd.concat(overlap_df)
        for row in overlap_df.to_dict(orient='records'):
            if row['id'] != self.id_: # Make sure we are not adding the sequence itself to the list, which overlaps with itself 100 percent.
                self.overlaps.append(Sequence(row['id'], metadata_df, genome_df=genome_df))

    def next(self, opposite:bool=False):
        # Get downstream gene on the relevant strand (either the same strand as the main ContextSequence, or the opposite strand).
        strand = self.strand if (not opposite) else self.opposite_strand
        return self.neighbors[strand]['left'] if (self.strand == '-') else self.neighbors[strand]['right'] 

    def overlap(self, seq:Sequence) -> int:
        '''Determine if the sequence overlaps with the input ContextSequence.'''
        # NOTE: This does not account for a case in which one gene completely contains the other.
        # If the stop of the overlapping gene lies in the bounds of the current gene...
        if (seq.stop > self.start) and (seq.stop < self.stop):
            return seq.stop - self.start
        # If the start of the overlapping gene lies in the bounds of the current gene... 
        elif (seq.start > self.start) and (seq.start < self.stop):
            return self.stop - seq.start
        else:
            return 0 
        
        # Overlap is present if the neighbor start is to the left of this gene's stop, so this value would be negative. 
        overlap = min(overlap, seq.start - self.stop)
        return abs(overlap)

    def distance(self, seq:Sequence):
        # TODO: Might be a good idea to make this signed, i.e. make it so that intergenic distance to the left is negative. 
        # Or maybe make the distance correspond to upstream versus downstream?
        if (seq.stop < self.start) and (seq.start < self.start): # Then the query gene is to the left of self.
            return self.start - seq.stop
        elif (seq.start > self.stop) and (seq.stop > self.stop): # Then the query gene is to the right of self.
            return seq.start - self.stop
        else: # TODO: Make sure this handles overlaps.
            None

    def is_downstream(self, seq) -> bool:
        '''Uses the start codon locations to determine if the input (query) Sequence object is 
        downstream of the self (reference) Sequence object.'''
        # Return None if the query gene is contained within the reference gene.
        if (seq.start >= self.start) and (seq.stop <= self.stop):
            return None

        # If the reference sequence is on the reverse strand, then the query sequence is downstream if the start
        # of the query sequence is to the left of the start of the reference sequence.
        if self.strand == '-':
            return seq.start <= self.start
        # If the reference sequence is on the forward strand, then the query sequence is downstream if the start
        # of the query sequence is to the right of the start of the reference sequence.
        elif self.strand == '+':
            return seq.start >= self.start

    def is_upstream(self, seq):
        '''Uses the start codon locations to determine if the input (query) Sequence object is 
        upstream of the self (reference) Sequence object.'''  
        # Return None if the query gene is contained within the reference gene.
        if (seq.start >= self.start) and (seq.stop <= self.stop):
            return None
 
        # If the reference sequence is on the reverse strand, then the query sequence is upstream if the start
        # of the query sequence is to the right of the start of the reference sequence.
        if self.strand == '-':
            return seq.start >= self.start
        # If the reference sequence is on the forward strand, then the query sequence is upstream if the start
        # of the query sequence is to the left of the start of the reference sequence.
        elif self.strand == '+':
            return seq.start <= self.start

    def extendable(self):
        # Check the case in which the gene is at the end of the contig.
        downstream = self.contig[:self.start] if self.strand == '-' else self.contig[self.stop:]
        if len(downstream) < 3: # Make sure there actually *is* a downstream region.
            return False
        return True

    def extend(self, verbose:bool=False):

        assert self.extendable(), f'ContextSequence.extend: No extension possible for ContextSequence {self.id_}.'

        downstream = self.contig[:self.start] if self.strand == '-' else self.contig[self.stop:]
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
        cseq_extended.ext_length = ext_length
        # Add new overlaps if the extended sequence now interferes with either downstream gene. 
        for seq in [self.next(), self.next(opposite=True)]:
            if (seq is not None) and (seq not in self.overlaps) and (cseq_extended.overlap(seq) > 0):
                cseq_extended.overlaps.append(seq)

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



# class FitnessBrowserSequence(ContextSequence):
#     '''Object for managing Fitness Browser data, including experimental fitness values and t-values.'''
#     # The columns in the fitness data which do not contain actual fitness values. 
#     non_experiment_cols = ['id', 'scaffold_id', 'pos', 'strand']
    
#     def __init__(self, id_:str, metadata_df:pd.DataFrame, fitness_df:pd.DataFrame, genome_df:pd.DataFrame=None, t_values_df:pd.DataFrame=None, low_memory:bool=True):

#         super().__init__(id_, metadata_df, genome_df=genome_df)

#         # Avoid copying the whole fitness_df if in low memory mode.
#         self._fitness_df = None if low_memory else copy.deepcopy(fitness_df)

#         self.load_fitness(fitness_df)

#         if t_values_df is None:
#             self.t_values = None
#         else:
#             # Load in the t-values for the specific gene.
#             t_values_df = t_values_df[t_values_df['id'] == self.id_]
#             # TODO: Look into why a locus might not have a t-value. 
#             # Not all loci seem to have computed t-values, so make sure to handle that case. 
#             self.t_values = None if (len(t_values_df) == 0) else t_values_df[self.fitness.columns]


#     def hits(self) -> int:
#         '''Return the number of fitness hits for the sequence.'''
#         return len(self.fitness)
    
#     def overlapping_hits(self) -> int:
#         # Get the locus IDs for all hits in the fitness data. 
#         hit_ids = self.fitness.index
#         # if np.issubdtype(hit_ids.dtype, np.number):
#         hit_ids = hit_ids[~pd.isnull(self.fitness.index)]
#         hit_ids = hit_ids.values[hit_ids.values != self.id_]
#         return len(hit_ids)

#     def mean(self, experiments:List[str]=None):
#         fitness = copy.deepcopy(self.fitness)
#         fitness = fitness.mean(axis=0)
#         if experiments is not None:
#             fitness = fitness[experiments]
#         return fitness.values.ravel()

#     def err(self, experiments:List[str]=None):
#         fitness = copy.deepcopy(self.fitness)
#         fitness = fitness.std(axis=0) / np.sqrt(self.hits())
#         if experiments is not None:
#             fitness = fitness[experiments]
#         return fitness.values
    
#     def significant(self, fitness_threshold:float=1, t_value_threshold:float=4):
#         '''Computes a "significance index" for a particular locus, which I am defining as the ratio
#         of experiments with significant phenotyes to all performed experiments.'''
#         meets_fitness_threshold = np.any(np.abs(self.mean()) > fitness_threshold)
#         meets_t_value_threshold = np.any(np.abs(self.t_values.values) > t_value_threshold)
#         return np.any(np.logical_and(meets_fitness_threshold, meets_t_value_threshold))

#     def load_fitness(self, fitness_df):
#         # Extract all hits from the fitness data. DON'T use the locus ID alone, as this will not work with the extended sequences. 
#         fitness_df = fitness_df[fitness_df.scaffold_id == self.scaffold_id]
#         fitness_df = fitness_df[fitness_df.strand == self.strand]
#         fitness_df = fitness_df[fitness_df['pos'] <= self.stop]
#         fitness_df = fitness_df[fitness_df['pos'] >= self.start]

#         # if 'id' in fitness_df.columns:
#         fitness_df = fitness_df.set_index('id') # Make sure to keep the ID in the DataFrame.
        
#         self.fitness = fitness_df[[c for c in fitness_df.columns if c not in FitnessBrowserSequence.non_experiment_cols]]
#         self.positions = fitness_df[['pos']]

#     def extend(self):

#         assert self._fitness_df is not None, 'FitnessBrowserSequence.extend: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
#         ext_fbseq = ContextSequence.extend(self) # Extend the sequence. copy.deepcopy is already called here.  
#         ext_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
#         return ext_fbseq

#     def extension(self):
#         '''Return only the extended region of an extended FitnessBrowserSequence, i.e. the region downstream of the original
#         stop codon.''' 
#         assert self._fitness_df is not None, 'FitnessBrowserSequence.extension: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
#         ext_only_fbseq = ContextSequence.extension(self) # Get the extended region. copy.deepcopy is already called here. 
#         ext_only_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
#         return ext_only_fbseq

#     def split(self):
#         assert self._fitness_df is not None, 'FitnessBrowserSequence.extension: FitnessBrowserSequence has been initialized in low memory mode. To extend the sequence, make sure low_memory=False.'
#         left_fbseq, right_fbseq = ContextSequence.split(self) # Get the split sequences. copy.deepcopy is already called here. 
#         left_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
#         right_fbseq.load_fitness(self._fitness_df) # Re-load the fitness data for the adjusted region. 
#         return left_fbseq, right_fbseq

#     def cofitness(self, fbseq): # -> Tuple[float, scipy.stats.LinregressResult]:
#         '''Compute the cofitness score between the two FitnessBrowserSequence objects. As defined in the Price et. al. 2018 paper,
#         the cofitness score is the absolute value of the r-value from regression.'''
#         linreg = scipy.stats.linregress(self.mean(), fbseq.mean())
#         return float(abs(linreg.rvalue)), linreg

