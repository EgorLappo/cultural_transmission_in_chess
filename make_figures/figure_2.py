import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(context='paper', style='ticks', palette='colorblind')

d = pd.read_csv('../data/csv/caissa_clean.csv', index_col=0, low_memory=False)
d = d[d.Year <= 2019]

fig, axs = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)

# plot number of games 
d1 = d.groupby('Year').pgn.count().reset_index()
d1['Games in dataset'] = d1.pgn

sns.lineplot(data=d1, x='Year', y='Games in dataset', ax=axs[0,0])

# plot number of players
d2 = []
for year in d.Year.unique():
    dyear = d[d.Year == year]
    unique_players = set(dyear.Black.unique()).union(set(dyear.White.unique()))
    d2.append({'Year': year, 'Unique players in dataset': len(unique_players)})

d2 = pd.DataFrame.from_records(d2)

sns.lineplot(data=d2, x='Year', y='Unique players in dataset', ax=axs[0,1])

axs[0,1].set_ylim(0, 20000)

# plot win rate 
d3 = d.groupby('Year').Result.mean().reset_index()

sns.lineplot(data=d3, x='Year', y='Result', ax=axs[1,0])

axs[1,0].set_ylim(-0.125, 0.125)
axs[1,0].axhline(y=0, color='r', linestyle='--')
axs[1,0].set_ylabel("Average outcome")

# plot game length 
d4 = d.groupby('Year').PlyCount.mean().reset_index()

sns.lineplot(data=d4, x='Year', y='PlyCount', ax=axs[1,1])

axs[1,1].set_ylabel("Average ply count")


axs[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
axs[0,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))

axs[0,0].set_xticks([1971,1980,1990,2000,2010,2019])
axs[0,1].set_xticks([1971,1980,1990,2000,2010,2019])
axs[1,0].set_xticks([1971,1980,1990,2000,2010,2019])
axs[1,1].set_xticks([1971,1980,1990,2000,2010,2019])

axs[0,0].text(-0.3, 1.1, "A", transform=axs[0,0].transAxes, fontsize=12, fontweight='bold', va='top')
axs[0,1].text(-0.3, 1.1, "B", transform=axs[0,1].transAxes, fontsize=12, fontweight='bold', va='top')
axs[1,0].text(-0.3, 1.1, "C", transform=axs[1,0].transAxes, fontsize=12, fontweight='bold', va='top')
axs[1,1].text(-0.3, 1.1, "D", transform=axs[1,1].transAxes, fontsize=12, fontweight='bold', va='top')


fig.savefig('../figures/figure_2.pdf')
