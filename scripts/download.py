import wget 
import gunzip

def download_swissprot(path:str) -> NoReturn:
    '''Downloads the current release of SwissProt from the UniProt FTP site.'''
    url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz'
    url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz'
    wget.download(url, path)


def download_selenoproteins(path:str) -> NoReturn:
    '''Downloads all known selenoproteins from UniProt.'''
    path = os.path.join(DATA_DIR, filename)
    # In order to include taxonomy information, need to download in CSV format. 
    url = 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cid%2Cgene_names%2Corganism_name%2Cdate_created%2Cversion%2Cannotation_score%2Cgo_p%2Cxref_refseq%2Cxref_kegg%2Clineage%2Clineage_ids%2Csequence&format=tsv&query=%28%28ft_non_std%3Aselenocysteine%29%29'
    wget.download(url, path)