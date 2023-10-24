# reuse figure 4 code
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(context='paper', style='ticks')

pal5 = ["#CC6677", "#88CCEE", "#332288", "#882255", "#DDCC77"]
pal6 = ["#CC6677", "#88CCEE", "#332288", "#882255", "#DDCC77",
        "#117733"]
pal9 = ["#CC6677", "#88CCEE", "#332288", "#882255", "#DDCC77",
        "#117733", "#44AA99", "#999933", "#AA4499"]

PLOT_RATIOS = True

strategies = ["queens_pawn_ply_2",
              "carokann_ply_5",
              "sicilian_najdorf_ply_11"]

strategies_clean_names = ["Queen's Pawn, Ply 2",
                          "Caro-Kann, Ply 5",
                          "Najdorf Sicilian, Ply 11"]

fbars = {}
Ns = {}

strategy = strategies[0]
strategy_clean_name = strategies_clean_names[0]

draws = pd.read_csv(
    "../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv(
    "../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_freqs = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index(
).melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        mp[i, j, :] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        fitness[i, j, :] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.01, axis=2)
mp_q95 = np.quantile(mp, 0.99, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.01, axis=2)
fitness_q95 = np.quantile(fitness, 0.99, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[
                str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[
                             str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[
                        str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[
                     str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[
                      str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(
    mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(
    fitness_q95, on=['Year', 'Move'])

# rename the move column values
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})

num_d = fitness[['Year', 'Move', 'fitness']].pivot(
    index='Year', columns='Move', values='fitness')
denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]

fbars[strategy] = fbar
Ns[strategy] = response_counts.sum(axis=1)

strategy = strategies[1]
strategy_clean_name = strategies_clean_names[1]

draws = pd.read_csv(
    "../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv(
    "../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_freqs = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index(
).melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        mp[i, j, :] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        fitness[i, j, :] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.01, axis=2)
mp_q95 = np.quantile(mp, 0.99, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.01, axis=2)
fitness_q95 = np.quantile(fitness, 0.99, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[
                str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[
                             str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[
                        str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[
                     str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[
                      str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(
    mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(
    fitness_q95, on=['Year', 'Move'])

# rename the move column values
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})


num_d = fitness[['Year', 'Move', 'fitness']].pivot(
    index='Year', columns='Move', values='fitness')
denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]

fbars[strategy] = fbar
Ns[strategy] = response_counts.sum(axis=1)

strategy = strategies[2]
strategy_clean_name = strategies_clean_names[2]

draws = pd.read_csv(
    "../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv(
    "../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_freqs = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index(
).melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        mp[i, j, :] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N, 39, 40000))

for i in range(N):
    for j in range(39):
        fitness[i, j, :] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.01, axis=2)
mp_q95 = np.quantile(mp, 0.99, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.01, axis=2)
fitness_q95 = np.quantile(fitness, 0.99, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[
                str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(
    x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[
                             str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[
                        str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[
                     str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[
                      str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(
    mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(
    fitness_q95, on=['Year', 'Move'])

# rename the move column values
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})

num_d = fitness[['Year', 'Move', 'fitness']].pivot(
    index='Year', columns='Move', values='fitness')
denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]

fbars[strategy] = fbar
Ns[strategy] = response_counts.sum(axis=1)

fig, axs = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)

for i, strategy in enumerate(strategies):
    strategy_clean_name = strategies_clean_names[strategies.index(strategy)]
    axs[0].plot(range(1980, 2019), fbars[strategy],
                label=strategy_clean_name, color=pal9[i])
    axs[1].plot(range(1980, 2019), np.array(Ns[strategy])[
                :-1]*np.array(fbars[strategy]), label=strategy_clean_name, color=pal9[i])

    axs[0].set_xlabel("Year")
    axs[1].set_xlabel("Year")
    axs[0].set_ylabel("$\\bar f$, mean fitness")
    axs[1].set_ylabel("$N_s(t)$, game sample size")

    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 12000)
    axs[0].set_xlim(1980, 2018)
    axs[1].set_xlim(1980, 2018)

    axs[0].set_xticks([1980, 1990, 2000, 2010, 2018])
    axs[1].set_xticks([1980, 1990, 2000, 2010, 2018])

    # legent on ax[1], top right, outside of plot
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

axs[0].text(-0.13, 1.0, 'A', transform=axs[0].transAxes,
            size='large', fontweight='bold', ha='right', va='top')
axs[1].text(-0.213, 1.0, 'B', transform=axs[1].transAxes,
            size='large', fontweight='bold', ha='right', va='top')

plt.savefig('../figures/figure_s5.pdf', bbox_inches='tight')
