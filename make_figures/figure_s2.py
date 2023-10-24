import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

strategies = ['queens_pawn_ply_2', 'carokann_ply_5', 'sicilian_najdorf_ply_11']
feature_names = ['all_count', 'top_count', 'top_win_rate', 'win_rate']
strategy_clean_names = {
    'queens_pawn_ply_2': 'Queens Pawn, ply 2',
    'carokann_ply_5': 'Caro-Kann, ply 5',
    'sicilian_najdorf_ply_11': 'Najdorf Sicilian, ply 11'
}
feature_clean_names = {
    'all_count': 'All game counts',
    'top_count': 'Top-50 game counts',
    'top_win_rate': 'Top-50 win rate',
    'win_rate': 'Win rate'
}

features = {}

for strategy in strategies:
    response_counts = pd.read_csv(f'../model/data/{strategy}/response_counts.csv').melt(
        id_vars='Year', var_name='Response', value_name='all_count')
    top_response_counts = pd.read_csv(f'../model/data/{strategy}/top_player_response_counts.csv').melt(
        id_vars='Year', var_name='Response', value_name='top_count')
    top_win_rates = pd.read_csv(f'../model/data/{strategy}/top_player_win_rates.csv').melt(
        id_vars='Year', var_name='Response', value_name='top_win_rate')
    win_rates = pd.read_csv(f'../model/data/{strategy}/win_rates.csv').melt(
        id_vars='Year', var_name='Response', value_name='win_rate')

    features[strategy] = top_response_counts.merge(
        top_win_rates).merge(win_rates).merge(response_counts)

# plot the three features for three strategies in a 3x3 grid
fig, axes = plt.subplots(4, 3, figsize=(
    15, 15), sharex=True, constrained_layout=True)

for j, strategy in enumerate(strategies):
    for i, feature in enumerate(feature_names):
        d = features[strategy]
        if strategy == 'queens_pawn_ply_2':
            d.win_rate = -d.win_rate
            d.top_win_rate = -d.top_win_rate
        sns.lineplot(data=features[strategy], x='Year',
                     y=feature, hue='Response', alpha=0.66, ax=axes[i, j])
        if i != 0:
            axes[i, j].get_legend().remove()

        axes[i, j].set_ylabel('')

axes[0, 0].set_ylabel('Count')
axes[1, 0].set_ylabel('Count')
axes[2, 0].set_ylabel('Win rate')
axes[3, 0].set_ylabel('Win rate')

for ax in axes[0, :]:
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

for ax in axes[-1, :]:
    ax.set_ylim(-1.05, 1.05)

for ax in axes[-2, :]:
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(1980,2019)
    ax.set_xticks([1980, 1990, 2000, 2010, 2019])

for j, strategy in enumerate(strategies):
    axes[0, j].text(0.5, 1.05, strategy_clean_names[strategy], fontweight='bold',
                    transform=axes[0, j].transAxes, size='xx-large', ha='center', va='baseline')

for i, feature in enumerate(feature_names):
    axes[i, 0].text(-0.3, 0.5, feature_clean_names[feature], transform=axes[i, 0].transAxes,
                    size='xx-large', fontweight='bold', rotation=90, ha='center', va='center')

for i, ax in enumerate(fig.axes):
    ax.text(-0.05, 1.05, chr(ord('@') + i + 1), transform=ax.transAxes,
            size='x-large', fontweight='bold', va='baseline', ha='center')

fig.align_labels()

fig.savefig('../figures/figure_s2.pdf', bbox_inches='tight')

