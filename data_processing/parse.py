import pandas as pd
import os

import chess
import chess.pgn

# WARNING: takes 10.5 hours to run on a laptop in this single-threaded form
# the parsing by chess.pgn.read_game is the bottleneck,
# but it ensures that the pgns are valid for all games 
# and leaves only the uniformly recorded mainline
n = 0
games = []

with open('../data/pgn/caissa.pgn', encoding='utf-8') as f:
    while True:
        game = chess.pgn.read_game(f)

        if game == None:
            break

        if n%100 == 0:
            print('processing game {n}'.format(n=n), end='\r')

        game_dict = {k:v for k,v in game.headers.items()}
        game_dict['pgn'] = str(game.mainline())
        games.append(game_dict)

        # checkpoints to save progress
        if n%100000 == 0: 
            d = pd.DataFrame.from_records(games)
            d.to_csv('../data/csv/caissa_full_'+str(n//100000)+'.csv')
            games = []
        
        n += 1

d = pd.DataFrame.from_records(games)
d.to_csv('../data/csv/caissa_full_'+str(n)+'.csv')

# concatenate the csvs
caissa_files = ['../data/csv/'+f for f in os.listdir('../data/csv/') if f.startswith('caissa_full_')]
caissa_parts = [pd.read_csv(f) for f in caissa_files]

d = pd.concat(caissa_parts)
d.to_csv('../data/csv/caissa_full.csv')