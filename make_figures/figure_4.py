import pandas as pd
import seaborn as sns
import matplotlib 
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(context='paper', style='ticks')

plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=11)
plt.rc('legend', fontsize=10) 

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

### *** PANELS A, B, C *** ###
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

if PLOT_RATIOS:
    num_d = fitness[['Year', 'Move', 'fitness']].pivot(
        index='Year', columns='Move', values='fitness')
    denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
    fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]
    fitness = num_d.div(fbar, axis=0).reset_index().melt(
        id_vars="Year", value_vars=moves, var_name="Move", value_name="fitness")

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_freqs = response_freqs[response_freqs['Move'] != 'other']

fig, axs = plt.subplots(3, 1, figsize=(3.3, 7), sharex=True)

sns.lineplot(data=response_freqs, x='Year', y='Frequency',
             hue='Move', ax=axs[0], alpha=0.8, palette=pal6)

axs[0].set_ylim(0, 1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean move frequency")
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))

axs[0].set_xlim(1980, 2018)
axs[0].set_xticks([1980, 1990, 2000, 2010, 2018])


for c, move in zip(pal6, moves):
    # plot the 5th and 95th percentiles using fill between
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move']
                        == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3, color=c)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q95', ax=axs[1], color='gray', alpha=0.3)

sns.lineplot(data=mp, x='Year', y='mean_prob', hue='Move',
             ax=axs[1], alpha=0.8, palette=pal6)

axs[1].set_ylim(0, 1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness',
             hue='Move', ax=axs[2], alpha=0.3, palette=pal6)
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in [
             'Nf6', 'd5'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette=pal6)
axs[2].set_ylabel("$f_i(p^i_t)/\\widebar{f}_t$")
axs[2].set_title("Frequency-dependent fitness")
axs[2].legend().remove()

# add bold letters to subfigures
axs[0].text(-0.225, 1, 'A', transform=axs[0].transAxes,
            size=12, weight='bold')
axs[1].text(-0.225, 1, 'B', transform=axs[1].transAxes,
            size=12, weight='bold')
axs[2].text(-0.225, 1, 'C', transform=axs[2].transAxes,
            size=12, weight='bold')

# make bold title
fig.suptitle(strategy_clean_name, fontsize=14, fontweight='bold')

fig.align_ylabels()

fig.subplots_adjust(hspace=0.35)

fig.savefig('../figures/figure_4_1.pdf', bbox_inches='tight')

### *** PANELS C, D, E *** ###
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

if PLOT_RATIOS:
    num_d = fitness[['Year', 'Move', 'fitness']].pivot(
        index='Year', columns='Move', values='fitness')
    denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
    fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]
    fitness = num_d.div(fbar, axis=0).reset_index().melt(
        id_vars="Year", value_vars=moves, var_name="Move", value_name="fitness")

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_freqs = response_freqs[response_freqs['Move'] != 'other']


fig, axs = plt.subplots(3, 1, figsize=(3.3, 7), sharex=True)

sns.lineplot(data=response_freqs, x='Year', y='Frequency',
             hue='Move', ax=axs[0], palette=pal5, alpha=0.8)

axs[0].set_ylim(0, 1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean move frequency")
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))

axs[0].set_xlim(1980, 2018)
axs[0].set_xticks([1980, 1990, 2000, 2010, 2018])

for c, move in zip(pal5, moves):
    # plot the 5th and 95th percentiles using fill between
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move']
                        == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3, color=c)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q95', ax=axs[1], color='gray', alpha=0.3)

sns.lineplot(data=mp, x='Year', y='mean_prob', hue='Move',
             ax=axs[1], palette=pal5, alpha=0.8)

axs[1].set_ylim(0, 1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness', hue='Move',
             ax=axs[2], alpha=0.3, palette=pal5)

aux_pal = [pal5[2], pal5[3]]
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in [
             'e5', 'exd5'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette=aux_pal)

axs[2].set_ylabel("$f_i(p^i_t)/\\widebar{f}_t$")
axs[2].set_title("Frequency-dependent fitness")
axs[2].legend().remove()


# add bold letters to subfigures
axs[0].text(-0.225, 1, 'D', transform=axs[0].transAxes,
            size=12, weight='bold')
axs[1].text(-0.225, 1, 'E', transform=axs[1].transAxes,
            size=12, weight='bold')
axs[2].text(-0.225, 1, 'F', transform=axs[2].transAxes,
            size=12, weight='bold')

fig.suptitle(strategy_clean_name, fontsize=14, fontweight='bold')

fig.align_ylabels()

fig.subplots_adjust(hspace=0.35)

fig.savefig('../figures/figure_4_2.pdf', bbox_inches='tight')

### *** PANELS G, H, I *** ###

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

if PLOT_RATIOS:
    num_d = fitness[['Year', 'Move', 'fitness']].pivot(
        index='Year', columns='Move', values='fitness')
    denum_d = response_counts.div(response_counts.sum(axis=1), axis=0)
    fbar = (num_d * denum_d).sum(axis=1).iloc[:-1]
    fitness = num_d.div(fbar, axis=0).reset_index().melt(
        id_vars="Year", value_vars=moves, var_name="Move", value_name="fitness")

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_freqs = response_freqs[response_freqs['Move'] != 'other']


fig, axs = plt.subplots(3, 1, figsize=(3.3, 7), sharex=True)

sns.lineplot(data=response_freqs, x='Year', y='Frequency',
             hue='Move', ax=axs[0], alpha=0.8, palette=pal9)

axs[0].set_ylim(0, 1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean move frequency")
axs[0].legend(loc='upper left', bbox_to_anchor=(0.997, 1.05))
# axs[0].legend().remove()

axs[0].set_xlim(1980, 2018)
axs[0].set_xticks([1980, 1990, 2000, 2010, 2018])

for c, move in zip(pal9, moves):
    # plot the 5th and 95th percentiles using fill between
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year',
                 y='q95', ax=axs[1], color='gray', alpha=0.3)
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move']
                        == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3, color=c)
sns.lineplot(data=mp, x='Year', y='mean_prob',
             hue='Move', ax=axs[1], alpha=0.8, palette=pal9)

axs[1].set_ylim(0, 1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness',
             hue='Move', ax=axs[2], alpha=0.3, palette=pal9)


aux_pal = [pal9[0], pal9[8]]
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in [
             'Bc4', 'h3'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette=aux_pal)

axs[2].set_xlim(1980, 2018)
axs[2].set_ylabel("$f_i(p^i_t)/\\widebar{f}_t$")
axs[2].set_title("Frequency-dependent fitness")
axs[2].legend().remove()

# add bold letters to subfigures
axs[0].text(-0.225, 1, 'G', transform=axs[0].transAxes,
            size=12, weight='bold')
axs[1].text(-0.225, 1, 'H', transform=axs[1].transAxes,
            size=12, weight='bold')
axs[2].text(-0.225, 1, 'I', transform=axs[2].transAxes,
            size=12, weight='bold')

handles, labels = axs[0].get_legend_handles_labels()

# fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1.05))

fig.suptitle(strategy_clean_name, fontsize=14, fontweight='bold')

fig.align_ylabels()

fig.subplots_adjust(hspace=0.35)

fig.savefig('../figures/figure_4_3.pdf', bbox_inches='tight')
