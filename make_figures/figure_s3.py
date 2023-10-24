import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

sns.set_theme(context='paper', style='ticks')

qp_fitness_data = pd.read_csv(
    "../model/model_results/queens_pawn_ply_2/mcmc_intervals_fitness.csv")
ck_fitness_data = pd.read_csv(
    "../model/model_results/carokann_ply_5/mcmc_intervals_fitness.csv")
naj_fitness_data = pd.read_csv(
    "../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_fitness.csv")

qp_responses = qp_fitness_data['response'].unique()
ck_responses = ck_fitness_data['response'].unique()
naj_responses = naj_fitness_data['response'].unique()

responses = [qp_responses, ck_responses, naj_responses]
tables = [qp_fitness_data, ck_fitness_data, naj_fitness_data]
strategies = ['Queen\'s Pawn, ply 2',
              'Caro-Kann, ply 5', 'Najdorf Sicilian, ply 11']

ylims = [(0.0, 0.4), (0.0, 0.8), (0.0, 0.6)]
nrows = [7, 6, 10]

fig, axs = plt.subplots(10, 3, figsize=(
    9, 12), sharex=True, tight_layout=True)

for x, sn in enumerate(strategies):
    # plot the rectangles, lines
    for y, r in enumerate(responses[x]):
        d = tables[x][tables[x]['response'] == r]
        for i, row in d.iterrows():
            axs[y, x].fill_between([row['lower_bp'], row['upper_bp']], [row['ll'], row['ll']], [
                                   row['hh'], row['hh']], alpha=0.4, color='orchid')
            axs[y, x].fill_between([row['lower_bp'], row['upper_bp']], [row['l'], row['l']], [
                                   row['h'], row['h']], alpha=0.7, color='purple')
        axs[y, x].plot(d['mid_bp'], d['m'], color='black', linewidth=2)
        axs[y, x].set_xscale('log')
        axs[y, x].set_title(r, loc='left', fontsize=14)
        axs[y, x].set_ylabel('$f_i(x)$', fontsize=12)

    axs[-1, x].set_xlabel('$x$', fontsize=12)

    # set y limits
    for j in range(10):
        axs[j, x].set_ylim(ylims[x])
        axs[j, x].set_xlim(0.0001, 1)

    # remove other axes
    for y in range(len(responses[x]), 10):
        axs[y, x].remove()

# yticks
for y in range(7):
    axs[y, 0].set_yticks([0, 0.1, 0.2, 0.3, 0.4])

for y in range(6):
    axs[y, 1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])

for y in range(10):
    axs[y, 2].set_yticks([0, 0.2, 0.4, 0.6])

# column labels A,B,C
for x in range(3):
    axs[0, x].text(-0.25, 1.6, chr(65+x), transform=axs[0, x].transAxes,
                   size=18, weight='bold')

# label columns by strategy
for x in range(3):
    axs[0, x].text(0.5, 1.6, strategies[x], transform=axs[0, x].transAxes,
                   size=14, weight='bold', ha='center')

fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig('../figures/figure_s3.pdf')
fig.savefig('../figures/figure_s3.png', dpi=500)
