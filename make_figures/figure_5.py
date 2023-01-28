import numpy as np
import pandas as pd
from re import split
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(context='paper', style='ticks', palette='colorblind')

d = pd.read_csv('../data/csv/caissa_player_games.csv', index_col=0)

f = lambda pgn: sum([m.strip().split() for m in split(r'\d+\.', pgn) if len(m) > 0],[])

def extract_first_plys(row):
    plys = {}
    pgn = f(row.pgn)
    for i in range(1,17):
        if i < len(pgn):
            plys['Ply'+str(i)] = ' '.join(pgn[:i])
        else:
            plys['Ply'+str(i)] = pd.NA
    
    return plys

ply_rows = []

for i, row in tqdm(d.iterrows()):
    ply_rows.append(extract_first_plys(row))

d_plys = pd.DataFrame.from_records(ply_rows)
d = d.merge(d_plys, left_index=True, right_index=True)

d['MinYear'] = d.groupby('Player').Year.transform(min)
d['DYear'] = d.Year - d.MinYear

def get_freqs(s):
    return s.value_counts() / s.value_counts().sum()

def entropy(fs): 
    fs = get_freqs(fs)
    return - fs.map(lambda x: x * np.log(x)).sum()

def heterozygosity(fs):
    fs = get_freqs(fs)
    return 1 - fs.map(lambda x: x * x).sum()

dgb_first_move = d[d.Pieces == 'White']\
    .groupby(['Player', 'DYear']).agg({'Elo' : 'mean', 'Result' : 'mean', 'Ply1': [entropy, heterozygosity]}).reset_index()
print('done with df 1')

dgb_e4_response = d[(d.Pieces == 'Black') & (d.Ply1 == 'e4')]\
    .groupby(['Player', 'DYear']).agg({'Elo' : 'mean', 'Result' : 'mean', 'Ply2': [entropy, heterozygosity]}).reset_index()
print('done with df 2')

dgb_first_move.columns = ['Player', 'DYear', 'Elo', 'Result', 'Ply1_Entropy', 'Ply1_Heterozygosity']
dgb_e4_response.columns = ['Player', 'DYear', 'Elo', 'Result', 'Ply2_Entropy', 'Ply2_Heterozygosity']

## MAKE PLOT 

fig, axs = plt.subplots(2, figsize = (3,4), constrained_layout=True) 

#sns.lineplot(data=dgb_first_move, x = 'DYear', y='Ply1_Entropy', ax = axs[0,0])
sns.lineplot(data=dgb_first_move, x = 'DYear', y='Ply1_Heterozygosity', ax = axs[0])
#sns.lineplot(data=dgb_e4_response, x = 'DYear', y='Ply2_Entropy', ax = axs[0,1])
sns.lineplot(data=dgb_e4_response, x = 'DYear', y='Ply2_Heterozygosity', ax = axs[1])

axs[0].set_title('Ply 1: First Move')
axs[1].set_title('Ply 2: e4 Response')

for i in range(2):
    axs[i].set_xlabel('Years in dataset')
    axs[i].set_ylabel('Heterozygosity')

fig.suptitle("Move choice diversity throughout the career")

fig.savefig('../figures/figure_5.pdf')
