import os 
import subprocess
from selenobot.files import CDHITFile, FASTAFile, MMseqsFile
import pandas as pd
import numpy as np
from selenobot.utils import default_output_path

class Kofamscan():

    def __init__(self, cmd_dir:str='/home/prichter/kofamscan/', tmp_dir:str='../data/tmp'):
        ''' 
        :param cmd_dir 
        :param tmp_dir: The temporary directory where hmmsearch results are. 
        '''
        self.cmd_dir = cmd_dir 
        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = tmp_dir 
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def run(self, input_path:str, output_path:str, prokaryote_only:bool=True, max_e_value:float=None, n_cpus:int=None) -> str:

        cmd = os.path.join(self.cmd_dir, 'bin', 'exec_annotation')
        cmd += f' {input_path} -o {output_path} --tmp-dir {self.tmp_dir}'

        if max_e_value is not None:
            cmd += f' --e-value {max_e_value}'
        if n_cpus is not None:
            cmd += f' --cpu {cpu_count}'

        if prokaryote_only:
            prokaryote_profiles_path = os.path.join(self.cmd_dir, 'db', 'profiles', 'prokaryote.hal')
            cmd += f' --profile {prokaryote_profiles_path}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)


# NOTE: Useful resource for submitting slurm jobs: https://docs.icer.msu.edu/BLAST_BLAST%2B_with_Multiple_Processors/
class BLAST():
    # https://open.oregonstate.education/computationalbiology/chapter/command-line-blast/
    # NOTE: HSP stands for High Scoring Pair.
    fields = ['qseqid', 'sseqid','pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'] # This is the default. 
    fields += ['qcovs', 'qcovhsp', 'qlen', 'slen'] # Some extra stuff which is helpful. 
    outfmt = '6 ' + ' '.join(fields)

    def __init__(self, cwd:str=os.getcwd()):
        self.cwd = cwd

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

    def run(self, query_path:str, subject_path:str, output_path:str, overwrite:bool=False, 
            max_high_scoring_pairs:int=None, 
            max_subject_sequences:int=None,
            make_database:bool=True,
            max_e_value:float=None,
            num_threads:int=4) -> str:
        '''Run the blastp program on the query and subject files.
        
        :param query_path
        :param subject_path
        :param overwrite 
        :param max_subject_sequences: For each query sequence, only report HSPs for the first specified different subject sequences.
        :param max_high_scoring_pairs: For each query-subject pair, only report the best n HSPs. In other words, this is the number of alignments for each 
            "hit", i.e. query-subject pair. 
        '''
        cmd = f'blastp -query {query_path} -out {output_path}' 
        if make_database:
            database_path = self.make_database(subject_path, overwrite=overwrite)
            cmd += f' -db {database_path}'
            cmd += f' -num_threads {num_threads}' # This is only relevant when searching against a database.  
        else:
            cmd += f' -subject {subject_path}'
        
        if max_high_scoring_pairs is not None:
            cmd += f' -max_hsps {max_high_scoring_pairs}'
        if max_subject_sequences is not None:
            cmd += f' -max_target_seqs {max_subject_sequences}'
        if max_e_value is not None:
            cmd += f' -evalue {max_e_value}'
        cmd += f' -outfmt \'{BLAST.outfmt}\'' # Use a custom output format. 

        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

        return output_path



class MMseqs():

    cleanup_paths = list()
    cleanup_paths += ['{base_path}_rep_seq.fasta']
    cleanup_paths += ['{base_path}_all_seqs.fasta']


    def __init__(self, tmp_dir:str='../data/tmp'):

        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = tmp_dir 
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def run(self, input_path:str, output_path:str, sequence_identity:float=0.2, overwrite:bool=False) -> str:

        cluster_path = output_path + '.tsv'

        if (not os.path.exists(cluster_path)) or overwrite:
            cmd = f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f'mv {output_path}_cluster.tsv {cluster_path}', shell=True, check=True)
        else:
            print(f'MMseqs.run: Using pre-saved clustering results at {cluster_path}')

        return cluster_path
            


class CDHIT():
    '''Class for managing the CD-HIT clustering of a group of amino acid sequences. Each instance handles
    the clustering and depreplication of a single FASTA file.
    
    CD-HIT user guide can be found here: http://www.bioinformatics.org/cd-hit/cd-hit-user-guide.pdf'''

    cleanup_files = list() # TODO

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

    def run(self, input_path:str, output_path:str, sequence_identity:float=0.8, overwrite:bool=False) -> str:
        '''Run the CD-HIT clustering tool on the data stored in the path attribute. CD-HIT prints out two files: output and output.clstr. 
        output contains the final clustered non-redundant sequences in FASTA format, while output.clstr has an information about the clusters 
        with its associated sequences.'''

        l = 5 # The minimum sequence length.
        c = sequence_identity 
        n = CDHIT.get_word_length(c)

        cluster_path = output_path + '.clstr'

        if (not os.path.exists(cluster_path)) or overwrite:
            # Run the CD-HIT command with the specified cluster parameters. 
            # CD-HIT can be installed using conda. conda install bioconda::cd-hit
            cmd = f'cd-hit -i {input_path} -o {output_path} -n {n} -c {c} -l {l}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        else:
            print(f'CDHIT.run: Using pre-saved clustering results at {output_path}')
        
        return cluster_path


class Clusterer():

    def __init__(self, tool:str='mmseqs', name:str='untitled', cwd:str=os.getcwd()):

        self.cwd = cwd
        self.name = name
        self.tool_name = tool
        self.tool = MMseqs() if (tool == 'mmseqs') else CDHIT()
        self.file_type = MMseqsFile if (tool == 'mmseqs') else CDHITFile
        self.input_path = os.path.join(self.cwd, self.name + '.fa')   
        self.base_path = os.path.join(self.cwd, self.name)   

    def cleanup(self):
        cleanup_paths = [path.format(base_path=self.base_path + f'.cluster') for path in self.tool.cleanup_paths]
        cleanup_paths += [path.format(base_path=self.base_path + f'.derep') for path in self.tool.cleanup_paths]
        cleanup_paths += [self.input_path]
        for path in cleanup_paths:
            if os.path.exists(path):
                print(f'Clusterer.cleanup: Removing output file at {path}')
                os.remove(path)


    def dereplicate(self, df:pd.DataFrame, overwrite:bool=False, sequence_identity:float=0.95) -> pd.DataFrame:
        '''Use a high sequence similarity threshold (specified by the c_dereplicate attribute) to dereplicate the proteins
        in the self.input_file. 
        '''
        # If the input DataFrame has already been clustered, make sure to drop the columns before re-clustering.
        df = df[[col for col in df.columns if col not in [f'{self.tool_name}_cluster', f'{self.tool_name}_representative']]]
        n = len(df) # Store the original number of sequences in the DataFrame. 

        # Don't include the descriptions with the FASTA file, because CD-HIT output files remove them anyway. 
        FASTAFile.from_df(df, add_description=False).write(self.input_path)

        output_path = default_output_path(self.base_path, op='derep')
        cluster_path = self.tool.run(self.input_path, output_path, sequence_identity=sequence_identity, overwrite=overwrite)
        cluster_df = self.file_type(cluster_path).to_df(reps_only=True) # Load in the cluster file and convert to a DataFrame.

        # Because CD-HIT filters out short sequences, the clstr_df might be smaller than the fasta_df. 
        df = cluster_df.merge(df, left_index=True, right_index=True, how='inner')
        print(f'Clusterer.dereplicate: Dereplication of clusters with {sequence_identity} similarity eliminated {n - len(df)} sequences from {self.name}. {len(df)} sequences remaining.')

        df = df.drop(columns=[f'{self.tool_name}_cluster', f'{self.tool_name}_representative']) # Don't need the cluster column after dereplication. 
        self.cleanup()
        return df 


    def cluster(self, df:pd.DataFrame, overwrite:bool=False, sequence_identity:float=0.3) -> pd.DataFrame:
        '''Use a lower sequence similarity threshold (specified by the c_cluster attribute) to cluster the proteins
        in the self.input_file. This is for use in the training-testing-validation split, to ensure that there is
        '''
        # If the input DataFrame has already been clustered, make sure to drop the columns before re-clustering.
        df = df[[col for col in df.columns if col not in [f'{self.tool_name}_cluster', f'{self.tool_name}_representative']]]

        FASTAFile.from_df(df, add_description=False).write(self.input_path)
 
        output_path = default_output_path(self.base_path, op='cluster') 
        cluster_path = self.tool.run(self.input_path, output_path, sequence_identity=sequence_identity, overwrite=overwrite)
        cluster_df = self.file_type(cluster_path).to_df(reps_only=False) # Load in the cluster file and convert to a DataFrame. Don't load only the representatives. 

        assert len(cluster_df) == len(df), f'Clusterer.cluster: There should be a cluster assigned to each sequence. len(clustr_df) = {len(cluster_df)} and len(df) = {len(df)}'

        # The DataFrame should now contain the clustering information. 
        df = cluster_df.merge(df, left_index=True, right_index=True, how='inner')
        self.cleanup()
        return df 


    def run(self, df:pd.DataFrame, overwrite:bool=True, cluster_sequence_identity:float=0.3, dereplicate_sequence_identity:float=0.95):
        df = self.dereplicate(df, sequence_identity=dereplicate_sequence_identity, overwrite=overwrite)
        df = self.cluster(df, sequence_identity=cluster_sequence_identity, overwrite=overwrite)
        return df # Return the dereplicated and clustered DataFrame. 
