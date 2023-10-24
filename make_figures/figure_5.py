import pandas as pd
import numpy as np

from tqdm import tqdm
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

NITER = 4*10_000
NTRAJ = 1000

pal9 = ["#CC6677", "#88CCEE", "#332288", "#882255", "#DDCC77",
        "#117733", "#44AA99", "#999933", "#AA4499"]


def f(x, cs, breakpoints):
    for i in range(len(breakpoints)):
        if x <= breakpoints[i]:
            return cs[i]
    return cs[-1]


def model_alphas(counts, features, fitness_values, breakpoints, betas):
    K = len(counts)
    freqs = counts/np.sum(counts)

    alphas = np.zeros(K)

    for k in range(K):
        alphas[k] = 1 + np.exp(np.dot(features[k, :], betas[k, :])) * \
            f(freqs[k], fitness_values[k, :], breakpoints[k, :]) * counts[k]

    return alphas


def counterfact_counts(K, moves, features, fitness_values, breakpoints, betas, steps=50, N=100000):
    seq = np.arange(0, 1, 1/steps)
    result = pd.DataFrame(data={"freq": seq})

    for k in range(K):
        probs = np.zeros(steps)

        for i, x in enumerate(seq):
            focal_count = N*x
            rest_count = (N - focal_count)/(K-1)
            counts = np.ones(K)*rest_count
            counts[k] = focal_count
            alphas = model_alphas(
                counts, features, fitness_values, breakpoints, betas)

            ps = np.random.dirichlet(alphas)

            probs[i] = ps[k] - x

        result[moves[k]] = probs

    return result


strategies = ["queens_pawn_ply_2",
              "carokann_ply_5", "sicilian_najdorf_ply_11"]
strategy_clean_names = {
    'queens_pawn_ply_2': 'Queens Pawn, ply 2',
    'carokann_ply_5': 'Caro-Kann, ply 5',
    'sicilian_najdorf_ply_11': 'Najdorf Sicilian, ply 11'
}
Ks = [7, 6, 10]

curves = {}

for strategy, K in zip(strategies, Ks):
    # betas are not used at all here
    beta_medians = pd.read_csv(
        f'../model/model_results/{strategy}/mcmc_intervals_beta.csv')
    beta_medians = beta_medians[['m']].values.reshape((K, 3), order='F')

    fitness_draw_df = pd.read_csv(
        f'../model/model_fits/{strategy}/fit.csv')
    fitness_draws = np.zeros((K, 4, NITER))
    for k in range(K):
        for i in range(4):
            fitness_draws[k, i, :] = \
                fitness_draw_df[f'fitness_values[{k+1},{i+1}]'].values

    fitness_medians = pd.read_csv(
        f'../model/model_results/{strategy}/mcmc_intervals_fitness.csv')
    fitness_medians = fitness_medians[['m']]\
        .values.reshape((K, 4), order='F')

    breakpoints = pd.read_csv(
        f'../model/model_fits/{strategy}/frequency_quantiles_by_strategy.csv')
    moves = breakpoints.iloc[:, 0].values
    breakpoints = breakpoints.iloc[:, 2:].values

    features = np.zeros((K, 3))

    ds = []

    for i in tqdm(range(NTRAJ), desc=strategy, total=NTRAJ):
        j = np.random.randint(NITER)
        fitness_samp = fitness_draws[:, :, j]

        dd = counterfact_counts(K, moves, features, fitness_samp, breakpoints, beta_medians)\
            .melt(id_vars=['freq'], var_name='move', value_name='prob')

        dd.rep = i

        ds.append(dd)

    d = pd.concat(ds)
    d = d.query('move != "other"')

    curves[strategy] = d

fig, ax = plt.subplots(1, 3, figsize=(
    14, 4), constrained_layout=True)

for i, strategy in enumerate(strategies):
    sns.lineplot(x='freq', y='prob', hue='move',
                 data=curves[strategy], errorbar=('ci', 98), ax=ax[i], alpha=0.66, palette=pal9)
    ax[i].set_xlabel('Move frequency ($x^i_t/N_t$)')
    ax[i].set_ylabel('Deviation from random choice ($p^i_t - x^i_t/N_t$)')
    ax[i].xaxis.label.set_fontsize(11)
    ax[i].yaxis.label.set_fontsize(11)

    ax[i].set_xlim(0, 1)
    # legend top right, outsize plot
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # horizontal line at 0
    ax[i].axhline(y=0, color='k', linestyle='--')

    ax[i].text(0.5, 1.05, strategy_clean_names[strategy], fontweight='bold',
               transform=ax[i].transAxes, size='large', ha='center', va='baseline')
    ax[i].text(-0.05, 1.05, chr(ord('@') + i + 1), transform=ax[i].transAxes,
               size='large', fontweight='bold', va='baseline', ha='center')

fig.savefig('../figures/figure_5.pdf', bbox_inches='tight')
