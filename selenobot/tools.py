import os 
import subprocess
from selenobot.files import CDHITFile, FASTAFile, MMseqsFile
import pandas as pd
import numpy as np 

class BLAST():
    # https://open.oregonstate.education/computationalbiology/chapter/command-line-blast/
    # NOTE: HSP stands for High Scoring Pair.
    
    def __init__(self, cwd:str=os.getcwd()):
        self.cwd = cwd
        self.output_format = '6'

    def make_database(self, path:str, overwrite:bool=False, verbose:bool=True):
        '''Make a database from the subject FASTA file to reduce the computational cost of searching.'''
        
        database_name = os.path.basename(path)
        database_name, _ = os.path.splitext(database_name)
        database_path = os.path.join(self.cwd, database_name)

        if (not os.path.exists(database_path)) or (not overwrite):
            cmd = f'makeblastdb -in {path} -out {database_path} -dbtype prot -title {database_name} -parse_seqids'
            print(f'BLAST.make_database: Creating database using {path} at {database_path}.')
            if verbose:
                print(cmd)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL) 
        else:
            print(f'BLAST.make_database: Using existing database at {database_path}.')
        return database_path

    def run(self, query_path:str, subject_path:str, overwrite:bool=False, verbose:bool=True, 
            max_high_scoring_pairs:int=1, 
            max_subject_sequences:int=1,
            make_database:bool=True) -> str:
        '''Run the blastp program on the query and subject files.
        
        :param query_path
        :param subject_path
        :param overwrite 
        :param verbose
        :param max_subject_sequences: For each query sequence, only report HSPs for the first specified different subject sequences.
        :param max_high_scoring_pairs: For each query/target pair, only report the best n HSPs.
        '''
        output_path = os.path.basename(query_path)
        output_path, _ = os.path.splitext(output_path)
        output_path = os.path.join(self.cwd, output_path + '.tsv') # Output should be a TSV file with the specified output format. 

        cmd = f'blastp -query {query_path} -out {output_path} -outfmt {self.output_format}' 
        if make_database:
            database_path = self.make_database(subject_path, overwrite=overwrite)
            cmd += f' -db {database_path}'
        else:
            cmd = f' -subject {subject_path}'
        cmd += f' -max_hsps {max_high_scoring_pairs} -max_target_seqs {max_subject_sequences}' # Add a few more parameters. 

        if verbose:
            print(cmd)

        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

        return output_path


# class MUSCLE():

#     def __init__(self, df:pd.DataFrame, name:str='untitled', cwd:str=os.getcwd()):
        
#         self.cwd = cwd
#         self.df = df 

#         self.input_path = os.path.join(self.cwd, f'{name}.fa')
#         self.output_path = os.path.join(self.cwd, f'{name}.afa') # This is a FASTA file format. 

#     def run(self):

#         # Write the DataFrame to a FASTA file so it can be used with MUSCLE. 
#         FASTAFile.from_df(self.df, add_description=False).write(self.input_path)

#         subprocess.run(f'muscle -in {self.input_path} -out {self.output_path}') 
#         return self.output_path

#     def cleanup(self):
#         os.remove(self.input_path)


class MMseqs():
    def __init__(self, cwd:str=None):

        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = os.path.join(cwd, 'tmp')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def run(self, input_path:str, output_path:str, sequence_identity:float=0.2, overwrite:bool=False, verbose:bool=True) -> str:

        if (not os.path.exists(output_path + '_cluster.tsv')) or overwrite:
            cmd = f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}'
            if verbose:
                print(cmd)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        else:
            print(f'MMseqs.run: Using pre-saved clustering results at {output_path}')

        return output_path + '_cluster.tsv'
            


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

    def __init__(self, cwd:str=None):
        pass

    def run(self, input_path:str, output_path:str, sequence_identity:float=0.8, overwrite:bool=False, verbose:bool=True) -> str:
        '''Run the CD-HIT clustering tool on the data stored in the path attribute. CD-HIT prints out two files: output and output.clstr. 
        output contains the final clustered non-redundant sequences in FASTA format, while output.clstr has an information about the clusters 
        with its associated sequences.'''

        l = 5 # The minimum sequence length.
        c = sequence_identity 
        n = CDHIT.get_word_length(c)

        if (not os.path.exists(output_path)) or overwrite:
            # Run the CD-HIT command with the specified cluster parameters. 
            # CD-HIT can be installed using conda. conda install bioconda::cd-hit
            cmd = f'cd-hit -i {input_path} -o {output_path} -n {n} -c {c} -l {l}'
            if verbose:
                print(cmd)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        else:
            print(f'CDHIT.run: Using pre-saved clustering results at {output_path}')
        
        return output_path + '.clstr'


class Clusterer():

    def __init__(self, tool:str='mmseqs', name:str='untitled', cwd:str=os.getcwd()):

        self.cwd = cwd
        self.name = name
        self.tool_name = tool
        self.tool = MMseqs(cwd=cwd) if (tool == 'mmseqs') else CDHIT()
        self.file_type = MMseqsFile if (tool == 'mmseqs') else CDHITFile
        self.input_path = os.path.join(self.cwd, self.name + '.fa')   


    def dereplicate(self, df:pd.DataFrame, overwrite:bool=False, sequence_identity:float=0.95) -> pd.DataFrame:
        '''Use a high sequence similarity threshold (specified by the c_dereplicate attribute) to dereplicate the proteins
        in the self.input_file. 
        '''
        # If the input DataFrame has already been clustered, make sure to drop the columns before re-clustering.
        df = df[[col for col in df.columns if col not in [f'{self.tool_name}_cluster', f'{self.tool_name}_representative']]]
        n = len(df) # Store the original number of sequences in the DataFrame. 

        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FASTAFile.from_df(df, add_description=False).write(self.input_path)

        output_path = os.path.join(self.cwd, f'dereplicate_{self.name}')
        cluster_file_path = self.tool.run(self.input_path, output_path, sequence_identity=sequence_identity, overwrite=overwrite)
        cluster_df = self.file_type(cluster_file_path).to_df(reps_only=True) # Load in the cluster file and convert to a DataFrame.

        # Because CD-HIT filters out short sequences, the clstr_df might be smaller than the fasta_df. 
        df = cluster_df.merge(df, left_index=True, right_index=True, how='inner')
        print(f'Clusterer.dereplicate: Dereplication of clusters with {sequence_identity} similarity eliminated {n - len(df)} sequences from {self.name}. {len(df)} sequences remaining.')

        df = df.drop(columns=[f'{self.tool_name}_cluster', f'{self.tool_name}_representative']) # Don't need the cluster column after dereplication. 
        # os.remove(self.input_path) # Remove the temporary FASTA file. 
        return df 


    def cluster(self, df:pd.DataFrame, overwrite:bool=False, sequence_identity:float=0.3) -> pd.DataFrame:
        '''Use a lower sequence similarity threshold (specified by the c_cluster attribute) to cluster the proteins
        in the self.input_file. This is for use in the training-testing-validation split, to ensure that there is
        '''
        # If the input DataFrame has already been clustered, make sure to drop the columns before re-clustering.
        df = df[[col for col in df.columns if col not in [f'{self.tool_name}_cluster', f'{self.tool_name}_representative']]]

        FASTAFile.from_df(df, add_description=False).write(self.input_path)
 
        output_path = os.path.join(self.cwd, f'cluster_{self.name}')
        cluster_file_path = self.tool.run(self.input_path, output_path, sequence_identity=sequence_identity, overwrite=overwrite)
        cluster_df = self.file_type(cluster_file_path).to_df(reps_only=False) # Load in the cluster file and convert to a DataFrame. Don't load only the representatives. 

        assert len(cluster_df) == len(df), f'Clusterer.cluster: There should be a cluster assigned to each sequence. len(clustr_df) = {len(cluster_df)} and len(df) = {len(df)}'

        # The DataFrame should now contain the clustering information. 
        df = cluster_df.merge(df, left_index=True, right_index=True, how='inner')

        # os.remove(self.input_path) # Remove the temporary FASTA file. 
        return df 


    def run(self, df:pd.DataFrame, overwrite:bool=True, cluster_sequence_identity:float=0.3, dereplicate_sequence_identity:float=0.95):
        df = self.dereplicate(df, sequence_identity=dereplicate_sequence_identity, overwrite=overwrite)
        df = self.cluster(df, sequence_identity=cluster_sequence_identity, overwrite=overwrite)
        return df # Return the dereplicated and clustered DataFrame. 
