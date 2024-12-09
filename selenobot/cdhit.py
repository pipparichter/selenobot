import os 
import subprocess
from selenobot.files import ClstrFile, FastaFile
import pandas as pd
import numpy as np 


class CdHit():
    '''Class for managing the CD-HIT clustering of a group of amino acid sequences. Each instance handles
    the clustering and depreplication of a single FASTA file.
    
    CD-HIT user guide can be found here: http://www.bioinformatics.org/cd-hit/cd-hit-user-guide.pdf'''

    def __init__(self, df:pd.DataFrame, name:str='tmp', cwd=os.getcwd()):
        ''' 
        :param df: A pandas DataFrame file containing sequences to cluster. .
        :param cwd: The working directory. This is where any output files will be written. 
        '''
        self.cwd = cwd
        self.df = df
        self.name = name

        self.dereplicated, self.clustered = False, False

        self.input_path = os.path.join(self.cwd, self.name + '.fa')
        
        self.dereplicate_output_path = os.path.join(self.cwd, f'dereplicate_{self.name}')
        self.cluster_output_path = os.path.join(self.cwd, f'cluster_{self.name}')
        
        self.c_dereplicate = 0.9 # Sequence similarity value for dereplication. 
        self.c_cluster = 0.8 # Sequence similarity value for clustering. 

    # def result(self) -> pd.DataFrame:
    #     return self.df

    def run(self, overwrite:bool=False):
        self.dereplicate(overwrite=overwrite)
        self.cluster(overwrite=overwrite)
        return self.df # Return the dereplicated and clustered DataFrame. 

    def dereplicate(self, overwrite:bool=False) -> pd.DataFrame:
        '''Use a high sequence similarity threshold (specified by the c_dereplicate attribute) to dereplicate the proteins
        in the self.input_file. 
        '''
        # Enforce order in the dereplicate -> cluster pipleline.
        assert not self.dereplicated, 'CdHit.dereplicate: Stored DataFrame has already been dereplicated.'
        assert not self.clustered, 'CdHit.dereplicate: Stored DataFrame has already been clustered.'

        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FastaFile.from_df(self.df, add_description=False).write(self.input_path)

        clstr_path = self._run(self.dereplicate_output_path, c=self.c_dereplicate, overwrite=overwrite) # Run CD-HIT and get the path of the output file. 
        clstr_df = ClstrFile(clstr_path).to_df(reps_only=True) # Load in the cluster file and convert to a DataFrame.

        # Because CD-HIT filters out short sequences, the clstr_df might be smaller than the fasta_df. 
        df = clstr_df.merge(self.df, left_index=True, right_index=True, how='inner')
        print(f'CdHit.dereplicate: Dereplication of clusters with {self.c_dereplicate} similarity eliminated {len(self.df) - len(df)} sequences from {self.name}.')
        print(df.columns)
        # df should now only contain the representative sequences. Store in the object.
        self.dereplicated = True # Mark the DataFrame as dereplicated. 
        self.df = df.drop(columns=['cluster']) # Don't need the cluster column after dereplication. 
        self.cleanup()

    def cluster(self, overwrite:bool=False) -> pd.DataFrame:
        '''Use a lower sequence similarity threshold (specified by the c_cluster attribute) to cluster the proteins
        in the self.input_file. This is for use in the training-testing-validation split, to ensure that there is
        '''
        # Enforce order in the dereplicate -> cluster pipleline.
        assert self.dereplicated, 'CdHit.cluster: Stored DataFrame has not yet been dereplicated.'
        assert not self.clustered, 'CdHit.cluster: Stored DataFrame has already been clustered.'
        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FastaFile.from_df(self.df, add_description=False).write(self.input_path)

        clstr_path = self._run(self.cluster_output_path, c=self.c_cluster, overwrite=overwrite) # Run CD-HIT and get the path of the output file. 
        clstr_df = ClstrFile(clstr_path).to_df(reps_only=False) # Load in the cluster file and convert to a DataFrame. 
        df = self.df.copy().merge(clstr_df, left_index=True, right_index=True, how='right')
        # The DataFrame should now contain the clustering information. 
        # df should now only contain the representative sequences. Store in the object.
        self.clustered = True # Mark the DataFrame as clustered.  
        self.df = df
        self.cleanup()

    def cleanup(self):
        '''Remove all files generated by this CdHit instance.'''
        os.remove(self.input_path) 
        # if os.path.exists(self.cluster_output_path):
        #     os.remove(self.cluster_output_path) 
        #     os.remove(self.cluster_output_path + '.clstr')
        # if os.path.exists(self.dereplicate_output_path):
        #     os.remove(self.dereplicate_output_path) 
        #     os.remove(self.dereplicate_output_path + '.clstr')  

    def _run(self, output_path:str, c:float=0.8, n:int=5, l:int=5, overwrite:bool=False) -> str:
        '''Run the CD-HIT clustering tool on the data stored in the path attribute. CD-HIT prints out two files: output and output.clstr. 
        output contains the final clustered non-redundant sequences in FASTA format, while output.clstr has an information about the clusters 
        with its associated sequences.'''

        if (not os.path.exists(output_path)) or overwrite:
            # Run the CD-HIT command with the specified cluster parameters. 
            # CD-HIT can be installed using conda. conda install bioconda::cd-hit
            subprocess.run(f'cd-hit -i {self.input_path} -o {output_path} -n {n} -c {c} -l {l}', shell=True, check=True, stdout=subprocess.DEVNULL)
        else:
            print(f'CdHit._run: Using pre-saved clustering results at {output_path}')
        
        return output_path + '.clstr'




