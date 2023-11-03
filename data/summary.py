
# def main(log_file_path=None):
#     '''Print the results of the setup procedure.'''
    
#     if log_file_path is not None: # Write the summary to a log file if specified. 
#         with open(log_file_path, 'w', encoding='UTF-8') as f:
#             sys.stdout = f

#     for file in os.listdir(DETECT_DATA_DIR):
#         path = os.path.join(DETECT_DATA_DIR, file)
#         print(f'[{file}]')
#         print('size:', csv_size(path))
#         sec_content = pd.read_csv(path, usecols=['label'])['label'].values.mean()
#         print('selenoprotein content:', np.round(sec_content, 3))
#         print()

#     print('[all_data.fasta]')
#     path = os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta')
#     seqs = fasta_seqs(path)
#     ids = fasta_ids(path)
#     print('total sequences:', len(seqs))
#     print('total selenoproteins:', len([i for i in ids if '[' in i]))
#     print(f'sequences of length >= {MIN_SEQ_LENGTH}:', np.sum(np.array([len(s) for s in seqs]) >= MIN_SEQ_LENGTH))
#     print(f'selenoproteins of length >= {MIN_SEQ_LENGTH - 1}:', len([i for i, s in zip(ids, seqs) if '[' in i and len(s) >= MIN_SEQ_LENGTH]))
#     print()

#     print('[all_data.fasta]')
#     path = os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta')
#     seqs = fasta_seqs(path)
#     ids = fasta_ids(path)
#     print('total sequences:', len(seqs))
#     print('total selenoproteins:', len([i for i in ids if '[' in i]))
#     print(f'sequences of length >= {MIN_SEQ_LENGTH}:', np.sum(np.array([len(s) for s in seqs]) >= MIN_SEQ_LENGTH))
#     print(f'selenoproteins of length >= {MIN_SEQ_LENGTH - 1}:', len([i for i, s in zip(ids, seqs) if '[' in i and len(s) >= MIN_SEQ_LENGTH]))
#     print()
