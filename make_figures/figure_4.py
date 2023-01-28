import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(context='paper', style='ticks', palette='colorblind')

strategies = ["queens_pawn_ply_2",
              "kings_pawn_ply_5",
              "sicilian_najdorf_ply_11"]

strategies_clean_names = ["Queen's Pawn, Ply 2",
                          "King's Pawn, Ply 5",
                          "Najdorf Sicilian, Ply 11"]

### *** PANELS A, B, C *** ###
strategy = strategies[0]
strategy_clean_name = strategies_clean_names[0]

draws = pd.read_csv("../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv("../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_counts = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index().melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        mp[i,j,:] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        fitness[i,j,:] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.05, axis=2)
mp_q95 = np.quantile(mp, 0.95, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.05, axis=2)
fitness_q95 = np.quantile(fitness, 0.95, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(fitness_q95, on=['Year', 'Move'])

# rename the move column values and (optionally) omit the "other" move
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_counts = response_counts[response_counts['Move'] != 'other']


fig, axs = plt.subplots(3, 1, figsize=(4,5), sharex = True, constrained_layout=True)

sns.lineplot(data=response_counts, x='Year', y='Frequency', hue='Move', ax=axs[0])

axs[0].set_ylim(0,1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean Move Frequency")
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))


for move in moves:
    # plot the 5th and 95th percentiles using fill between
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move'] == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q95', ax=axs[1], color='gray', alpha=0.3)
    
sns.lineplot(data=mp, x='Year', y='mean_prob', hue='Move', ax=axs[1], palette='colorblind')

axs[1].set_ylim(0,1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness', hue='Move', ax=axs[2], alpha=0.3, palette='colorblind')
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in ['Nf6', 'd5'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette='colorblind')
axs[2].set_ylabel("$f_i(x^i_t/N_t)$")
axs[2].set_title("Fitness of a strategy")
axs[2].legend().remove()

# add bold letters to subfigures
axs[0].text(-0.2, 1.1, 'A', transform=axs[0].transAxes, size=12, weight='bold')
axs[1].text(-0.2, 1.1, 'B', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(-0.2, 1.1, 'C', transform=axs[2].transAxes, size=12, weight='bold')

fig.suptitle("{}".format(strategy_clean_name), fontsize=12)

fig.savefig('../figures/figure_4_1.png', dpi=300, bbox_inches='tight')

### *** PANELS C, D, E *** ###
strategy = strategies[1]
strategy_clean_name = strategies_clean_names[1]

draws = pd.read_csv("../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv("../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_counts = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index().melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        mp[i,j,:] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        fitness[i,j,:] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.05, axis=2)
mp_q95 = np.quantile(mp, 0.95, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.05, axis=2)
fitness_q95 = np.quantile(fitness, 0.95, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(fitness_q95, on=['Year', 'Move'])

# rename the move column values and (optionally) omit the "other" move
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_counts = response_counts[response_counts['Move'] != 'other']


fig, axs = plt.subplots(3, 1, figsize=(4,5), sharex = True, constrained_layout=True)

sns.lineplot(data=response_counts, x='Year', y='Frequency', hue='Move', ax=axs[0], palette='colorblind')

axs[0].set_ylim(0,1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean Move Frequency")
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))


for move in moves:
    # plot the 5th and 95th percentiles using fill between
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move'] == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q95', ax=axs[1], color='gray', alpha=0.3)
    
sns.lineplot(data=mp, x='Year', y='mean_prob', hue='Move', ax=axs[1], palette='colorblind')

axs[1].set_ylim(0,1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness', hue='Move', ax=axs[2], alpha=0.3, palette='colorblind')

p = sns.color_palette('colorblind', n_colors=4)
aux_pal = [p[0], p[-1]]
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in ['Bb5','d4'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette=aux_pal)

axs[2].set_ylabel("$f_i(x^i_t/N_t)$")
axs[2].set_title("Fitness of a strategy")
axs[2].legend().remove()


# add bold letters to subfigures
axs[0].text(-0.2, 1.1, 'D', transform=axs[0].transAxes, size=12, weight='bold')
axs[1].text(-0.2, 1.1, 'E', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(-0.2, 1.1, 'F', transform=axs[2].transAxes, size=12, weight='bold')

fig.suptitle("{}".format(strategy_clean_name), fontsize=12)

fig.savefig('../figures/figure_4_2.png', dpi=300, bbox_inches='tight')

### *** PANELS G, H, I *** ###

strategy = strategies[2]
strategy_clean_name = strategies_clean_names[2]

draws = pd.read_csv("../model/model_fits/{}/fit.csv".format(strategy), index_col=0)

response_counts = pd.read_csv("../model/data/{}/response_counts.csv".format(strategy), index_col=0)
moves = list(response_counts.columns)
response_counts = response_counts.div(response_counts.sum(axis=1), axis=0).reset_index().melt(id_vars="Year", value_vars=moves, var_name="Move", value_name="Frequency")

mp_cols = [c for c in draws.columns if "meanprobs" in c]
fitness_cols = [c for c in draws.columns if "fitness[" in c]

N = len(moves)

mp = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        mp[i,j,:] = draws[mp_cols[i*39+j]]

fitness = np.zeros((N,39,20000))

for i in range(N):
    for j in range(39):
        fitness[i,j,:] = draws[fitness_cols[i*39+j]]

mp_means = np.mean(mp, axis=2)
mp_q5 = np.quantile(mp, 0.05, axis=2)
mp_q95 = np.quantile(mp, 0.95, axis=2)

fitness_means = np.mean(fitness, axis=2)
fitness_q5 = np.quantile(fitness, 0.05, axis=2)
fitness_q95 = np.quantile(fitness, 0.95, axis=2)

# make dataframes out of arrays
mp_means = pd.DataFrame(mp_means.T, columns=[str(x) for x in range(N)])
mp_means['Year'] = 1980 + np.array(range(39))
mp_means = pd.melt(mp_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='mean_prob')

mp_q5 = pd.DataFrame(mp_q5.T, columns=[str(x) for x in range(N)])
mp_q5['Year'] = 1980 + np.array(range(39))
mp_q5 = pd.melt(mp_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

mp_q95 = pd.DataFrame(mp_q95.T, columns=[str(x) for x in range(N)])
mp_q95['Year'] = 1980 + np.array(range(39))
mp_q95 = pd.melt(mp_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

fitness_means = pd.DataFrame(fitness_means.T, columns=[str(x) for x in range(N)])
fitness_means['Year'] = 1980 + np.array(range(39))
fitness_means = pd.melt(fitness_means, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='fitness')

fitness_q5 = pd.DataFrame(fitness_q5.T, columns=[str(x) for x in range(N)])
fitness_q5['Year'] = 1980 + np.array(range(39))
fitness_q5 = pd.melt(fitness_q5, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q5')

fitness_q95 = pd.DataFrame(fitness_q95.T, columns=[str(x) for x in range(N)])
fitness_q95['Year'] = 1980 + np.array(range(39))
fitness_q95 = pd.melt(fitness_q95, id_vars=['Year'], value_vars=[str(x) for x in range(N)], var_name='Move', value_name='q95')

# merge the dataframes
mp = pd.merge(mp_means, mp_q5, on=['Year', 'Move']).merge(mp_q95, on=['Year', 'Move'])
fitness = pd.merge(fitness_means, fitness_q5, on=['Year', 'Move']).merge(fitness_q95, on=['Year', 'Move'])

# rename the move column values and (optionally) omit the "other" move
mp['Move'] = mp['Move'].map({str(x): moves[x] for x in range(N)})
fitness['Move'] = fitness['Move'].map({str(x): moves[x] for x in range(N)})

mp = mp[mp['Move'] != 'other']
fitness = fitness[fitness['Move'] != 'other']
response_counts = response_counts[response_counts['Move'] != 'other']


fig, axs = plt.subplots(3, 1, figsize=(3.3,5), sharex = True, constrained_layout=True)

sns.lineplot(data=response_counts, x='Year', y='Frequency', hue='Move', ax=axs[0])

axs[0].set_ylim(0,1)
axs[0].set_ylabel("Frequency")
axs[0].set_title("Mean Move Frequency")
# axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
axs[0].legend().remove()

for move in moves:
    # plot the 5th and 95th percentiles using fill between
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q5', ax=axs[1], color='gray', alpha=0.3)
    sns.lineplot(data=mp[mp['Move'] == move], x='Year', y='q95', ax=axs[1], color='gray', alpha=0.3)
    axs[1].fill_between(mp[mp['Move'] == move]['Year'], mp[mp['Move'] == move]['q5'], mp[mp['Move'] == move]['q95'], alpha=0.3)
sns.lineplot(data=mp, x='Year', y='mean_prob', hue='Move', ax=axs[1])
    
axs[1].set_ylim(0,1)
axs[1].set_ylabel("Probability")
axs[1].set_title("Mean move choice probability")
axs[1].legend().remove()

sns.lineplot(data=fitness, x='Year', y='fitness', hue='Move', ax=axs[2], alpha=0.3)

p = sns.color_palette('colorblind', n_colors=9)
aux_pal = [p[0], p[8]]
sns.lineplot(data=fitness[fitness.Move.map(lambda x: x in ['Bc4','h3'])], x='Year', y='fitness', hue='Move', ax=axs[2], palette=aux_pal)

axs[2].set_xlim(1980,2018)
axs[2].set_ylabel("$f_i(x^i_t/N_t)$")
axs[2].set_title("Fitness of a strategy")
axs[2].legend().remove()

# add bold letters to subfigures
axs[0].text(-0.2, 1.1, 'G', transform=axs[0].transAxes, size=12, weight='bold')
axs[1].text(-0.2, 1.1, 'H', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(-0.2, 1.1, 'I', transform=axs[2].transAxes, size=12, weight='bold')

handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.04,0.91))

fig.suptitle("{}".format(strategy_clean_name), fontsize=12)

fig.savefig('../figures/figure_4_3.png', dpi=300, bbox_inches='tight')
