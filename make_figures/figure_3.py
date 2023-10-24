import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')

sns.set_theme(context='paper', style='ticks', palette='colorblind')


def make_ply_streamplot_ax(ply, ax, lab, title, palette, col='Moves', thresh=0.02):
    ply_int = ply.reset_index(drop=True).fillna(0).copy()  # int for internal
    ply_int['Count'] = ply_int['Count'] / \
        ply_int.groupby('Year')['Count'].transform('sum')

    mean_freqs = ply_int.groupby(col).agg({'Count': 'mean'}).Count
    rare_moves = [m for m in list(ply_int[col]) if mean_freqs[m] < thresh]
    ply_int[col] = ply_int[col].replace({m: 'other' for m in rare_moves})
    ply_int = ply_int.groupby(['Year', col]).agg(
        {'Count': 'sum', 'Result': 'mean'}).reset_index()

    ply_piv = ply_int.pivot(index='Year', columns=col,
                            values='Count').reset_index()
    ply_piv = ply_piv.fillna(0)

    # trick to have the order of colors match the order of labels
    data = -np.array(ply_piv[ply_piv.columns[1:]]).T
    labels = ply_piv.columns[1:]

    # change captions to be short
    labels = [l.split()[-1] for l in list(labels)]

    ax.stackplot(list(ply_piv.Year), data,
                 labels=labels, alpha=1, baseline='zero', colors=palette)
    ax.legend(bbox_to_anchor=(1.04, 1.04), loc="upper left")
    # ax.set_title('Opening strategy relative frequencies')
    ax.set_xlabel('Year')
    ax.set_ylabel('Fraction of Games')
    # "matching order" trick continues
    ax.set_ylim(-1, 0)
    ax.set_xlim(1971, 2019)
    ax.set_xticks([1971, 1980, 1990, 2000, 2010, 2019])
    ax.set_yticks([-1 + i*0.2 for i in range(6)],
                  [f"{i*0.2:2.2}" for i in range(6)])

    ax.text(-0.23, 1.1, lab, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.set_title(title, fontweight="bold")


ply1 = pd.read_csv('../data/csv/caissa_counts_by_ply1.csv', index_col=0)

ply4 = pd.read_csv('../data/csv/caissa_counts_by_ply3.csv', index_col=0)
sicilian = ply4[(ply4.Ply1 == 'e4') & (ply4.Ply2 == 'c5')]

ply11 = pd.read_csv('../data/csv/caissa_counts_by_ply11.csv', index_col=0)
ndorf = ply11[(ply11.Ply1 == 'e4') & (ply11.Ply2 == 'c5') & (ply11.Ply3 == 'Nf3') & (ply11.Ply4 == 'd6') & (ply11.Ply5 == 'd4') & (
    ply11.Ply6 == 'cxd4') & (ply11.Ply7 == 'Nxd4') & (ply11.Ply8 == 'Nf6') & (ply11.Ply9 == 'Nc3') & (ply11.Ply10 == 'a6')]

ply7 = pd.read_csv('../data/csv/caissa_counts_by_ply7.csv', index_col=0)
qgd = ply7[(ply7.Ply1 == 'd4') & (ply7.Ply2 == 'd5') & (ply7.Ply3 == 'c4') & (
    ply7.Ply4 == 'e6') & (ply7.Ply5 == 'Nc3') & (ply7.Ply6 == 'Nf6')]

fig, axs = plt.subplots(4, figsize=(3, 12), sharex=False)

pal4 = ["#CC6677", "#332288", "#88CCEE", "#DDDDDD"]
pal5 = ["#CC6677", "#332288", "#88CCEE", "#882255", "#DDDDDD"]
pal10 = ["#CC6677", "#332288", "#88CCEE", "#882255", "#DDCC77",
         "#117733", "#44AA99", "#999933", "#AA4499", "#DDDDDD"]

make_ply_streamplot_ax(
    ply1, axs[0], 'A', 'Starting Position, ply 1', palette=pal5)
make_ply_streamplot_ax(
    sicilian, axs[1], 'B', 'Sicilian Defense, ply 3', palette=pal4)
make_ply_streamplot_ax(
    qgd, axs[2], 'C', 'Queen\'s Gambit Declined, ply 7', palette=pal4)
make_ply_streamplot_ax(
    ndorf, axs[3], 'D', 'Najdorf Sicilian, ply 11', palette=pal10)

fig.subplots_adjust(hspace=.4)

fig.align_ylabels()

fig.savefig('../figures/figure_3.pdf', bbox_inches='tight')
