import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from re import split
from collections import defaultdict as dd

parse_pgn_string = lambda pgn: sum([m.strip().split() for m in split(r'\d+\.', pgn) if len(m) > 0],[])

def add_plys(row):
    plys = parse_pgn_string(row.pgn)
    for i, ply in enumerate(plys):
        if i <= 24:
            row[f'Ply_{i+1}'] = ply
    return row

def summarize_responses(df, other_thresh=0.02):
    responses = df.Response.value_counts()
    response_freq = responses / responses.sum() 
    rare_responses = response_freq[response_freq < other_thresh].index
    df['Response'] = df.Response.map(lambda x: 'other' if x in rare_responses else x)
    return(df)

# takes a dataframe and a dict from year to list of players
def select_top_players(d, top_players_dict, ply):
    chunks = []
    for y, ps in top_players_dict.items():
        chunk = d[d.Year == y]
        if ply % 2 == 1:
            chunk = chunk[chunk.White.isin(ps)]
        else:
            chunk = chunk[chunk.Black.isin(ps)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def prepare_dataset(d, strategy, strategy_name, other_thresh=0.02, top_players_dict=None):
    # subset the data
    strategy_list = parse_pgn_string(strategy)
    k = len(strategy_list) + 1
    i = 1 if k % 2 == 1 else 0 

    df = d[d.PlyCount >= k+1]
    df = df[df.pgn.str.startswith(strategy)]

    df['Response'] = df.pgn.apply(lambda x: split(strategy, x)[1].strip().split()[i].strip())

    df = summarize_responses(df, other_thresh)

    print(f'creating directory data/{strategy_name}...')
    os.makedirs(f'data/{strategy_name}', exist_ok=True)

    print('making data tables...')
    print(f'selected {df.shape[0]} games...')
    
    response_counts = df.groupby('Year').Response.value_counts().reset_index(name='count')
    win_rates = df.groupby(['Year','Response']).Result.mean().reset_index(name='win_rate')

    response_counts = response_counts.pivot(index='Year', columns='Response', values='count').fillna(0)
    win_rates = win_rates.pivot(index='Year', columns='Response', values='win_rate').fillna(0)
    
    print('writing data tables...')
    response_counts.to_csv(f'data/{strategy_name}/response_counts.csv')
    win_rates.to_csv(f'data/{strategy_name}/win_rates.csv')

    if top_players_dict:
        d_top_players = select_top_players(df, top_players_dict, k)

        print('making data tables for top players...')
        top_player_response_counts = d_top_players.groupby('Year').Response.value_counts().reset_index(name='count')
        top_player_win_rates = d_top_players.groupby(['Year','Response']).Result.mean().reset_index(name='win_rate')

        top_player_response_counts = top_player_response_counts.pivot(index='Year', columns='Response', values='count').fillna(0)
        top_player_win_rates = top_player_win_rates.pivot(index='Year', columns='Response', values='win_rate').fillna(0)

        # deal with the fact that some responses are not present in the top players
        for c in response_counts.columns:
            if c not in top_player_response_counts.columns:
                top_player_response_counts[c] = 0
                top_player_win_rates[c] = 0

        print('writing data tables for top players...')
        top_player_response_counts.to_csv(f'data/{strategy_name}/top_player_response_counts.csv')
        top_player_win_rates.to_csv(f'data/{strategy_name}/top_player_win_rates.csv')

    print(f'done with strategy {strategy_name}')
    
    return df

### *** GENERATE FEATURE TABLES FOR ALL STRATEGIES *** ###

d = pd.read_csv('../data/csv/caissa_clean.csv', index_col=0, low_memory=False)

# leave only relevant columns
d = d[['Year', 'Black', 'White', 'Result', 'WhiteElo', 'BlackElo', 'PlyCount', 'pgn']]
# only recent games, pre-covid
d = d[d['Year'] >= 1980]
d = d[d['Year'] <= 2019]

N_top_players = 50

white_players = d[['Year', 'White', 'WhiteElo', 'Result']]
black_players = d[['Year', 'Black', 'BlackElo', 'Result']]
white_players.columns = ['Year', 'Player', 'Elo', 'Result']
black_players.columns = ['Year', 'Player', 'Elo', 'Result']
black_players.Result = - black_players.Result

players = pd.concat([white_players,  black_players], ignore_index=True).drop_duplicates().groupby(['Year', 'Player']).agg({'Elo':'mean', 'Result': 'mean'}).reset_index()

top_players = players.groupby('Year').apply(lambda x: x.nlargest(N_top_players, 'Elo')).Player.reset_index().drop('level_1', axis=1)
player_dict = dd(list)
for i, row in top_players.iterrows():
    player_dict[row.Year].append(row.Player)

prepare_dataset(d, '1. d4', 'queens_pawn_ply_2', top_players_dict=player_dict)
prepare_dataset(d, '1. e4 c6 2. d4 d5', 'carokann_ply_5', top_players_dict=player_dict)
prepare_dataset(d, '1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6', 'sicilian_najdorf_ply_11', top_players_dict=player_dict)
