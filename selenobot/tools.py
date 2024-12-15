import os 
import subprocess
from selenobot.files import CDHITFile, FASTAFile, MMseqsFile
import pandas as pd
import numpy as np 


class MUSCLE():

    def __init__(self, df:pd.DataFrame, name:str='untitled', cwd:str=os.getcwd()):
        
        self.cwd = cwd
        self.df = df 

        self.input_path = os.path.join(self.cwd, f'{name}.fa')
        self.output_path = os.path.join(self.cwd, f'{name}.afa') # This is a FASTA file format. 

    def run(self):

        # Write the DataFrame to a FASTA file so it can be used with MUSCLE. 
        FASTAFile.from_df(self.df, add_description=False).write(self.input_path)

        subprocess.run(f'muscle -in {self.input_path} -out {self.output_path}') 
        return self.output_path

    def cleanup(self):
        os.remove(self.input_path)


class MMseqs():
    def __init__(self, df:pd.DataFrame, name:str='untitled', cwd=os.getcwd()):

        self.cwd = cwd
        self.df = df 
        self.name = name

        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = os.path.join(cwd, 'tmp')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        self.input_path = os.path.join(self.cwd, f'{name}.fa')
        self.output_path = os.path.join(self.cwd, f'{name}_cluster.tsv')

    def run(self, min_seq_id:float=0.2, overwrite:bool=False) -> pd.DataFrame:
        ''' 
        :param min_seq_id: A minimum sequence identity defined as the equivalent similarity score (derived from a training set) of the local alignment (including gap penalties) 
            divided by the maximum of the lengths of the two locally aligned sequence segments. 
        :param e: A maximum E-value threshold computed according to the gap-corrected Karlin-Altschul statistics using the ALP library.
        :param c: A minimum coverage between 0 and 1, which is defined by the number of aligned residue pairs divided by 
            the maximum of the length of query and target sequences alnRes/max(qLen,tLen). 
        '''
        if (not os.path.exists(self.output_path)) or overwrite:
            FASTAFile.from_df(self.df, add_description=False).write(self.input_path)
            cmd = f'mmseqs easy-cluster {self.input_path} {self.name} tmp --min-seq-id {min_seq_id}'
            subprocess.run(cmd, shell=True, check=True)
        else:
            print(f'MMseqs.run: Using pre-saved clustering results at {self.output_path}')

        cluster_df = MMseqsFile(self.output_path).to_df()
        self.df = self.df.merge(cluster_df, left_index=True, right_index=True)
        return self.df
            


class CDHIT():
    '''Class for managing the CD-HIT clustering of a group of amino acid sequences. Each instance handles
    the clustering and depreplication of a single FASTA file.
    
    CD-HIT user guide can be found here: http://www.bioinformatics.org/cd-hit/cd-hit-user-guide.pdf'''

    @staticmethod
    def get_word_length(c:float):
        '''Get the word length for the filter, which is dependent on the specified similarity.'''
        if c >= 0.7:
            return 5 
        elif c >= 0.6:
            return 4 
        elif c >= 0.5:
            return 3
        elif c >= 0.4:
            return 2 
        else:
            raise Exception(f'CDHIT.get_word_length: Specified c value {c} is too low for the short-word filter.')

    def __init__(self, df:pd.DataFrame, name:str='tmp', cwd=os.getcwd(), c_cluster:float=0.8):
        ''' 
        :param df: A pandas DataFrame file containing sequences to cluster.
        :param name: The name of the CD-HIT clustering job.
        :param cwd: The working directory. This is where any output files will be written. 
        '''
        self.cwd = cwd
        # If the input DataFrame has already been clustered, make sure to drop the columns before re-clustering.
        self.df = df[[col for col in df.columns if col not in ['cdhit_cluster', 'cdhit_representative']]]
        self.name = name

        self.dereplicated, self.clustered = False, False

        self.input_path = os.path.join(self.cwd, self.name + '.fa')
        
        self.dereplicate_output_path = os.path.join(self.cwd, f'dereplicate_{self.name}')
        self.cluster_output_path = os.path.join(self.cwd, f'cluster_{self.name}')
        
        self.c_dereplicate = 0.9 # Sequence similarity value for dereplication. 
        self.c_cluster = c_cluster # Sequence similarity value for clustering. 

    def run(self, overwrite:bool=False, dereplicate:bool=True):
        if dereplicate:
            self._dereplicate(overwrite=overwrite)
        self._cluster(overwrite=overwrite)
        return self.df # Return the dereplicated and clustered DataFrame. 

    def _dereplicate(self, overwrite:bool=False) -> pd.DataFrame:
        '''Use a high sequence similarity threshold (specified by the c_dereplicate attribute) to dereplicate the proteins
        in the self.input_file. 
        '''
        # Enforce order in the dereplicate -> cluster pipleline.
        assert not self.dereplicated, 'CDHIT.dereplicate: Stored DataFrame has already been dereplicated.'
        assert not self.clustered, 'CDHIT.dereplicate: Stored DataFrame has already been clustered.'

        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FASTAFile.from_df(self.df, add_description=False).write(self.input_path)

        clstr_path = self._run(self.dereplicate_output_path, c=self.c_dereplicate, overwrite=overwrite) # Run CD-HIT and get the path of the output file. 
        clstr_df = CDHITFile(clstr_path).to_df(reps_only=True) # Load in the cluster file and convert to a DataFrame.

        # Because CD-HIT filters out short sequences, the clstr_df might be smaller than the fasta_df. 
        df = clstr_df.merge(self.df, left_index=True, right_index=True, how='inner')
        print(f'CDHIT.dereplicate: Dereplication of clusters with {self.c_dereplicate} similarity eliminated {len(self.df) - len(df)} sequences from {self.name}. {len(df)} sequences remaining.')
        # df should now only contain the representative sequences. Store in the object.
        self.dereplicated = True # Mark the DataFrame as dereplicated. 

        self.df = df.drop(columns=['cdhit_cluster', 'cdhit_representative']) # Don't need the cluster column after dereplication. 
        self.cleanup()

    def _cluster(self, overwrite:bool=False) -> pd.DataFrame:
        '''Use a lower sequence similarity threshold (specified by the c_cluster attribute) to cluster the proteins
        in the self.input_file. This is for use in the training-testing-validation split, to ensure that there is
        '''
        # Enforce order in the dereplicate -> cluster pipleline.
        # assert self.dereplicated, 'CDHIT.cluster: Stored DataFrame has not yet been dereplicated.'
        assert not self.clustered, 'CDHIT.cluster: Stored DataFrame has already been clustered.'
        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FASTAFile.from_df(self.df, add_description=False).write(self.input_path)

        clstr_path = self._run(self.cluster_output_path, c=self.c_cluster, overwrite=overwrite) # Run CD-HIT and get the path of the output file. 
        clstr_df = CDHITFile(clstr_path).to_df(reps_only=False) # Load in the cluster file and convert to a DataFrame. 
        assert len(clstr_df) == len(self.df), f'CDHIT.cluster: There should be a cluster assigned to each remaining sequence. len(clstr_df) = {len(clstr_df)} and len(self.df) = {len(self.df)}'

        df = clstr_df.merge(self.df, left_index=True, right_index=True, how='inner')
        # The DataFrame should now contain the clustering information. 
        # df should now only contain the representative sequences. Store in the object.
        self.clustered = True # Mark the DataFrame as clustered.  
        self.df = df
        self.cleanup()

    def cleanup(self):
        '''Remove all files generated by this CDHIT instance.'''
        os.remove(self.input_path) 

    def _run(self, output_path:str, c:float=0.8, l:int=5, overwrite:bool=False) -> str:
        '''Run the CD-HIT clustering tool on the data stored in the path attribute. CD-HIT prints out two files: output and output.clstr. 
        output contains the final clustered non-redundant sequences in FASTA format, while output.clstr has an information about the clusters 
        with its associated sequences.'''

        if (not os.path.exists(output_path)) or overwrite:
            # Run the CD-HIT command with the specified cluster parameters. 
            # CD-HIT can be installed using conda. conda install bioconda::cd-hit
            n = CDHIT.get_word_length(c)
            cmd = f'cd-hit -i {self.input_path} -o {output_path} -n {n} -c {c} -l {l}'
            print(cmd) 
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        else:
            print(f'CDHIT._run: Using pre-saved clustering results at {output_path}')
        
        return output_path + '.clstr'




