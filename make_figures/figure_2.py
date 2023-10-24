import pandas as pd
import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(context='paper', style='ticks', palette='colorblind')

d = pd.read_csv('../data/csv/caissa_clean.csv', index_col=0, low_memory=False)
d = d[d.Year <= 2019]

fig, axs = plt.subplots(2, 2, figsize=(5.5, 4), constrained_layout=True)

# plot number of games
d1 = d.groupby('Year').pgn.count().reset_index()
d1['Games in dataset'] = d1.pgn

sns.lineplot(data=d1, x='Year', y='Games in dataset', ax=axs[0, 0])
axs[0, 0].set_ylabel("Game count")

# plot number of players
d2 = []
for year in d.Year.unique():
    dyear = d[d.Year == year]
    unique_players = set(dyear.Black.unique()).union(set(dyear.White.unique()))
    d2.append({'Year': year, 'Player count': len(unique_players)})

d2 = pd.DataFrame.from_records(d2)

sns.lineplot(data=d2, x='Year', y='Player count', ax=axs[0, 1])

axs[0, 1].set_ylim(0, 20000)

# plot game length
d3 = d.groupby('Year').PlyCount.mean().reset_index()
sns.lineplot(data=d3, x='Year', y='PlyCount', ax=axs[1, 1])
axs[1, 1].set_ylabel("Mean ply count")

d4 = d.groupby('Year').Result.value_counts()
d4.name = 'Count'
d4 = d4.reset_index()
d4 = d4.pivot(index='Year', columns='Result', values='Count')
d4 = d4.fillna(0)
d4 = d4.div(d4.sum(axis=1), axis=0)
d4.columns = ['Black wins', 'Draw', 'White wins']
d4 = d4.melt(ignore_index=False).reset_index()
d4.columns = ["Year", "Outcome", "Frequency"]

sns.lineplot(data=d4, x='Year', y='Frequency', hue='Outcome', ax=axs[1, 0])

axs[1, 0].set_ylim(0, 1)
# no legend title, position top right
axs[1, 0].legend(title=None, loc='upper right', bbox_to_anchor=(1, 1))


# formatting
axs[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
axs[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))

axs[0, 0].set_xticks([1971, 1980, 1990, 2000, 2010, 2019])
axs[0, 1].set_xticks([1971, 1980, 1990, 2000, 2010, 2019])
axs[1, 0].set_xticks([1971, 1980, 1990, 2000, 2010, 2019])
axs[1, 1].set_xticks([1971, 1980, 1990, 2000, 2010, 2019])

axs[0, 0].set_yticks([0, 50000, 100000])

axs[0, 0].text(-0.245, 1.0, "A", transform=axs[0, 0].transAxes,
               fontsize=12, fontweight='bold', va='top', ha='right')
axs[0, 1].text(-0.225, 1.0, "B", transform=axs[0, 1].transAxes,
               fontsize=12, fontweight='bold', va='top', ha='right')
axs[1, 0].text(-0.245, 1.0, "C", transform=axs[1, 0].transAxes,
               fontsize=12, fontweight='bold', va='top', ha='right')
axs[1, 1].text(-0.225, 1.0, "D", transform=axs[1, 1].transAxes,
               fontsize=12, fontweight='bold', va='top', ha='right')

fig.align_ylabels()

fig.savefig('../figures/figure_2.pdf', bbox_inches='tight')
