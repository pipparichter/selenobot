import unittest
from selenobot.utils import dataframe_from_fasta, ncbi_parser
from selenobot.extend import * 
import re
from parameterized import parameterized
# Files for testing were downloaded from NCBI. 

# Load the data from the E. coli genome, for use in testing.
NT_DF = dataframe_from_fasta('./data/NC_000913.3_cds.fna', parser=ncbi_parser)
NT_DF_RC = dataframe_from_fasta('./data/NC_000913.3_cds_reverse_complement.fna', parser=ncbi_parser)
AA_DF = dataframe_from_fasta('./data/NC_000913.3_cds.faa', parser=ncbi_parser)
# Clean up the DataFrames by setting the index and dropping rows with missing values.
for df in [NT_DF, NT_DF_RC, AA_DF]:
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates('gene_id', keep='first', inplace=True)
    df.set_index('gene_id', inplace=True)
# self.rc_nt_df, self.nt_df, self.aa_df = rc_nt_df, nt_df, aa_df 
GENOME = dataframe_from_fasta('./data/NC_000913.3_genome.fna', parser=None).seq.values[0]

GENE_IDS = set(NT_DF.index).intersection(set(AA_DF.index)).intersection(set(NT_DF_RC.index))
GENE_IDS = list(GENE_IDS)[:100] # Probably don't need to test on *all* of the genes. 
SEC_GENE_IDS = ['fdnG', 'fdoG', 'fdhF']

def truncate(gene_id:str):

    row = AA_DF.loc[gene_id] # Access the information from the amino acid DataFrame.
    start, stop = int(row.start), int(row.stop) #  - 1 # Shift because the start index is non-inclusive.
    orientation = row.orientation

    sec_idx = row.seq.index('U')
    if row.orientation == '+':
        stop = start - 1 + ((sec_idx + 1) * 3) # This is the new end of the truncated sequence. 
        # print('truncate: U codon is', GENOME[stop - 3:stop])
    elif row.orientation == '-':
        start = stop - ((sec_idx + 1) * 3)  + 1
        # print('truncate: U codon is', get_reverse_complement(GENOME[start - 1:start + 2]))

    return start, stop


class ExtendTests(unittest.TestCase):
    '''Tests for the extend utilities.'''

    def setUp(self):
        self.maxDiff = None

    @parameterized.expand(GENE_IDS)
    def test_get_reverse_complement(self, gene_id:str):
        '''Test the reverse complement function.'''
        nt_seq = NT_DF.loc[gene_id]['seq']
        nt_seq_rc = NT_DF_RC.loc[gene_id]['seq']

        self.assertEqual(nt_seq, get_reverse_complement(nt_seq_rc))
        self.assertEqual(nt_seq_rc, get_reverse_complement(nt_seq))

    @parameterized.expand(GENE_IDS)
    def test_get_start_codon(self, gene_id:str):

        row = NT_DF.loc[gene_id]
        start, stop, orientation = row['start'], row['stop'], row['orientation']
        start, stop = int(start) - 1, int(stop)

        start_codon = get_start_codon(GENOME[start:stop], orientation=orientation)
        self.assertTrue(start_codon in START_CODONS[orientation])

    @parameterized.expand(GENE_IDS)
    def test_get_start_codon(self, gene_id:str):

        row = NT_DF.loc[gene_id]
        start, stop, orientation = row['start'], row['stop'], row['orientation']
        start, stop = int(start) - 1, int(stop)

        stop_codon = get_stop_codon(GENOME[start:stop], orientation=orientation)
        self.assertTrue(stop_codon in STOP_CODONS[orientation])

    @parameterized.expand(SEC_GENE_IDS)
    def test_extend(self, sec_gene_id:str):
        '''Test the extend function on truncated selenoproteins.'''

        row = AA_DF.loc[sec_gene_id]
        original_start, original_stop, orientation = int(row.start), int(row.stop), row['orientation']
        trunc_start, trunc_stop = truncate(sec_gene_id)

        trunc_length, original_length = trunc_stop - trunc_start, original_stop - original_start
        ext_results = extend(start=trunc_start, stop=trunc_stop, contig=GENOME, orientation=orientation, verbose=False)

        self.assertEqual(trunc_length + ext_results['nt_ext'], original_length)
        self.assertEqual(GENOME[original_start - 1:original_stop], ext_results['seq'])

    @parameterized.expand(GENE_IDS + SEC_GENE_IDS) # Test on both selenoproteins and non-selenoproteins. 
    def test_translate(self, gene_id:str):
        '''Test to make sure the translate function correctly converts the nucleotide string to amino acids.'''
        row = AA_DF.loc[gene_id]
        start, stop, orientation = int(row.start), int(row.stop), row['orientation']

        aa_seq = translate(GENOME[start - 1:stop], orientation=orientation)
        ref_aa_seq = row.seq # Extract the correct amino acid sequence from the NCBI FASTA file. 

        self.assertEqual(len(aa_seq), len(ref_aa_seq))
        self.assertEqual(aa_seq, ref_aa_seq)

    # When testing extension on randomly-truncated proteins, need to make sure the truncation preserves the 
    # reading frame, and need to disable the check that enforces a stop codon at the end of the original sequence. 
    # @parameterized.expand(GENE_IDS)
    # def test_extend(self, gene_id:str):
    #     original_length


if __name__ == '__main__':
    unittest.main()