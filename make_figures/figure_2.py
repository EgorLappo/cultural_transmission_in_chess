import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker
#import mpltern
import seaborn as sns

sns.set_theme(context='paper', style='ticks', palette='colorblind')

d = pd.read_csv('../data/csv/caissa_clean.csv', index_col=0, low_memory=False)
d = d[d.Year <= 2019]

fig, axs = plt.subplots(4, figsize=(3, 8), constrained_layout=True)

# plot number of games 
d1 = d.groupby('Year').pgn.count().reset_index()
d1['Games in dataset'] = d1.pgn

sns.lineplot(data=d1, x='Year', y='Games in dataset', ax=axs[0])
axs[0].set_ylabel("Game count")

# plot number of players
d2 = []
for year in d.Year.unique():
    dyear = d[d.Year == year]
    unique_players = set(dyear.Black.unique()).union(set(dyear.White.unique()))
    d2.append({'Year': year, 'Player count': len(unique_players)})

d2 = pd.DataFrame.from_records(d2)

sns.lineplot(data=d2, x='Year', y='Player count', ax=axs[1])

axs[1].set_ylim(0, 20000)

# plot game length 
d3 = d.groupby('Year').PlyCount.mean().reset_index()
sns.lineplot(data=d3, x='Year', y='PlyCount', ax=axs[2])
axs[2].set_ylabel("Average ply count")

d4 = d.groupby('Year').Result.value_counts()
d4.name = 'Count'
d4 = d4.reset_index()
d4 = d4.pivot(index='Year', columns='Result', values='Count')
d4 = d4.fillna(0)
d4 = d4.div(d4.sum(axis=1), axis=0)
d4.columns = ['Black wins', 'Draw', 'White wins']
d4 = d4.melt(ignore_index=False).reset_index()
d4.columns = ["Year", "Outcome", "Frequency"]

sns.lineplot(data=d4, x='Year', y='Frequency', hue='Outcome', ax=axs[3])

axs[3].set_ylim(0,1)
# no legend title, position top right
axs[3].legend(title=None, loc='upper right', bbox_to_anchor=(1, 1))

# axs[3].remove()
# tern_ax = fig.add_subplot(4,1,4, projection='ternary')

# # calculate proportions of entries in Result column per year
# vc = d.groupby('Year').Result.value_counts()
# vc.name = 'Count'
# vc = vc.reset_index()
# vc = vc.pivot(index='Year', columns='Result', values='Count')
# vc = vc.fillna(0)
# vc = vc.div(vc.sum(axis=1), axis=0)
# vc.columns = ['Black wins', 'Draw', 'White wins']

# # plot ternary diagram
# tern_ax.plot(vc['Draw'], vc['Black wins'], vc['White wins'])
# tern_ax.set_tlabel('Draw')
# tern_ax.set_llabel('Black wins')
# tern_ax.set_rlabel('White wins')
# tern_ax.set_title('Game outcome proportions')

# position = 'tick1'
# tern_ax.taxis.set_label_position(position)
# tern_ax.laxis.set_label_position(position)
# tern_ax.raxis.set_label_position(position)

# tern_ax.text(vc.iloc[0,0], vc.iloc[0,1], vc.iloc[0,2], '1971', va='top', ha='right')
# tern_ax.text(vc.iloc[-1,0], vc.iloc[-1,1], vc.iloc[-1,2], '2019', va='top', ha='right')


## formatting
axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))

axs[0].set_xticks([1971,1980,1990,2000,2010,2019])
axs[1].set_xticks([1971,1980,1990,2000,2010,2019])
axs[2].set_xticks([1971,1980,1990,2000,2010,2019])


axs[0].text(-0.4, 1.0, "A", transform=axs[0].transAxes, fontsize=12, fontweight='bold', va='top')
axs[1].text(-0.4, 1.0, "B", transform=axs[1].transAxes, fontsize=12, fontweight='bold', va='top')
axs[2].text(-0.4, 1.0, "C", transform=axs[2].transAxes, fontsize=12, fontweight='bold', va='top')
axs[3].text(-0.4, 1.0, "D", transform=axs[3].transAxes, fontsize=12, fontweight='bold', va='top')

fig.align_ylabels()

fig.savefig('../figures/figure_2.pdf', bbox_inches='tight')
