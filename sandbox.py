from utils import data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = '/home/prichter/Documents/selenobot/data/'
fasta = data.read_fasta(data_dir + 'uniprot_081123_sec.fasta')

df = data.fasta_to_df(fasta)

df['sec_content'] = df['seq'].apply(lambda s : s.count('U'))

fig, ax = plt.subplots(1)#, figsize=(15, 10))

counts, edges, bars = plt.hist(df['sec_content'].values, log=True, bins=np.arange(19))
plt.bar_label(bars, counts.astype(int))

ax.set_xticks(edges)
ax.set_xticklabels(edges.astype(int))
ax.set_ylabel('log(count)')
ax.set_xlabel('selenocysteine content')

ax.set_title('Selenocysteine content in known selenoproteins')

# sns.histplot(data=df, x='sec_content', ax=ax, log_scale=(False, True))
fig.savefig('/home/prichter/Documents/selenobot/data/sec_content_histogram.png')
