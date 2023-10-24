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

qp_beta_data = pd.read_csv(
    "../model/model_results/queens_pawn_ply_2/mcmc_intervals_beta.csv")
ck_beta_data = pd.read_csv(
    "../model/model_results/carokann_ply_5/mcmc_intervals_beta.csv")
naj_beta_data = pd.read_csv(
    "../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_beta.csv")

qp_responses = qp_fitness_data['response'].unique()
ck_responses = ck_fitness_data['response'].unique()
naj_responses = naj_fitness_data['response'].unique()

qp_beta_data['response'] = list(qp_responses)*3
ck_beta_data['response'] = list(ck_responses)*3
naj_beta_data['response'] = list(naj_responses)*3

qp_beta_data['variable'] = [
    '$\\beta_{win}$']*7 + ['$\\beta_{top50-win}$']*7 + ['$\\beta_{top50-freq}$']*7
ck_beta_data['variable'] = [
    '$\\beta_{win}$']*6 + ['$\\beta_{top50-win}$']*6 + ['$\\beta_{top50-freq}$']*6
naj_beta_data['variable'] = [
    '$\\beta_{win}$']*10 + ['$\\beta_{top50-win}$']*10 + ['$\\beta_{top50-freq}$']*10

responses = [qp_responses, ck_responses, naj_responses]
tables = [qp_beta_data, ck_beta_data, naj_beta_data]
strategies = ['Queen\'s Pawn, ply 2',
              'Caro-Kann, ply 5', 'Najdorf Sicilian, ply 11']
variables = ['$\\beta_{top50-freq}$',
             '$\\beta_{top50-win}$', '$\\beta_{win}$',]


nrows = [7, 6, 10]

fig, axs = plt.subplots(10, 3, figsize=(
    9, 12), sharex=True, tight_layout=False)

for x, sn in enumerate(strategies):
    # plot the rectangles, lines
    for y, r in enumerate(responses[x]):
        d = tables[x][tables[x]['response'] == r]
        for z, v in enumerate(variables):
            yy = z/2
            m = d[d['variable'] == v]['m'].values[0]
            l = d[d['variable'] == v]['l'].values[0]
            h = d[d['variable'] == v]['h'].values[0]
            ll = d[d['variable'] == v]['ll'].values[0]
            hh = d[d['variable'] == v]['hh'].values[0]
            axs[y, x].axvline(color='gray', alpha=0.8, linewidth=1)
            axs[y, x].plot([l, h], [yy, yy], color='black', linewidth=5)
            axs[y, x].plot([ll, hh], [yy, yy], color='black', linewidth=3)
            axs[y, x].scatter(m, yy, color='black', s=50, zorder=10)

        axs[y, x].set_title(r, loc='left', fontsize=14)
        axs[y, x].set_ylim(-0.2, 1.2)
        axs[y, x].set_yticks([0, 0.5, 1])
        axs[y, x].set_yticklabels(variables)

    axs[-1, x].set_xlabel('Value', fontsize=12)

for y in range(6):
    axs[y, 1].set_xlim(-0.3, 0.3)

# column labels A,B,C
for x in range(3):
    axs[0, x].text(-0.35, 1.6, chr(65+x), transform=axs[0, x].transAxes,
                   size=18, weight='bold')

# label columns by strategy
for x in range(3):
    axs[0, x].text(0.5, 1.6, strategies[x], transform=axs[0, x].transAxes,
                   size=14, weight='bold', ha='center')

fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig('../figures/figure_s4.pdf')
fig.savefig('../figures/figure_s4.png', dpi=500)
