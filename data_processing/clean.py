import numpy as np
import pandas as pd
import os
from re import split
from unidecode import unidecode

d = pd.read_csv('../data/csv/caissa_full.csv', index_col=0)

def resolve_naming_difference(d, column1, column2):
    d.loc[d[column1].isna(),column1] = d.loc[d[column1].isna(),column2]
    return d.drop(column2, axis=1) 

d = resolve_naming_difference(d, 'WhiteElo', 'WhiteELO')
d = resolve_naming_difference(d, 'BlackElo', 'BlackELO')
d = resolve_naming_difference(d, 'ECO', 'Eco')
d = resolve_naming_difference(d, 'PlyCount', 'Plycount')
d = resolve_naming_difference(d, 'EventCountry','Eventcountry')
d = resolve_naming_difference(d, 'EventDate','Eventdate')
d = resolve_naming_difference(d, 'EventRounds', 'Eventrounds')
d = resolve_naming_difference(d, 'EventType', 'Eventtype')
d = resolve_naming_difference(d, 'SetUp', 'Setup')

# `?` seems to be a default convention for a missing value in chess
d[d == '?'] = np.NaN
d[d == '??'] = np.NaN
d[['WhiteElo', 'BlackElo']] = d[['WhiteElo','BlackElo']].replace(0.0, np.NaN)
d = d[d.WhiteType != 'program']
d = d[d.BlackType != 'program']
d = d[d.Termination != 'unterminated']

d = d.dropna(subset=['pgn', 'WhiteElo','BlackElo', 'ECO', 'Result'])

d = d.drop(columns = ['Unnamed: 0', 'Source', 'Source1', 'Source2', 'Source3', 'Source4', 'SourceDate', 'Sourcedate', 'Time', 'SectionMarker', 'FileName', 'Remark', 'Owner', 'Input', 'Classes', 'Termination', 'SectionMarker', 'PresId', 'LiveChessVersion', 'ID', 'Annotator', 'Comment', 'NIC','Time','SetUp','Game', 'BlackType','WhiteType','EventRounds', 'Board', 'EventCategory'])

string_cols = ['Event', 'Round', 'EventType', 'EventCountry', 'White', 'Black', 'Result', 'ECO', 'Opening', 'Variation', 'FEN', 'WhiteTitle', 'BlackTitle', 'WhiteCountry', 'BlackCountry', 'WhiteTeam', 'BlackTeam', 'WhiteTeamCountry','BlackTeamCountry', 'TimeControl', 'BlackClock', 'WhiteClock', 'pgn']
num_cols = ['WhiteElo', 'BlackElo', 'WhiteFideId', 'BlackFideId', 'PlyCount']
date_cols = ['Date', 'EventDate']


date_entries = ['Year', 'Month', 'Day']
event_date_entries = ['EventYear', 'EventMonth','EventDay']
d_dates = pd.DataFrame.from_records(list(d.Date.astype('str').map(lambda row: {date_entries[i]:v for i, v in enumerate(row.strip().split('.'))})))

d.reset_index(inplace=True)
d = d.merge(d_dates, left_index=True, right_index=True)
d[date_entries] = d[date_entries].replace('??', np.NaN)
d[date_entries] = d[date_entries].replace('????', np.NaN)

d[date_cols + string_cols] = d[date_cols+string_cols].astype('string')
d[num_cols+date_entries] = d[num_cols+date_entries].astype('float64')

d = d[date_cols + date_entries + string_cols + num_cols]

d = d[d.Result != '*']
d = d[d.Year > 1970]
d = d.dropna(subset=['Year', 'White', 'Black', 'Result'])

d.Result = d.Result.replace({'1-0': '1', '0-1': '-1', '1/2-1/2':'0'})
d.Result = d.Result.astype('int32')

d.reset_index(drop=True, inplace=True)

# to work with players it's important to make sure that the names are unique
# this is NOT the case for so many people
# below is a hacky way to resolve this with some information loss (although it's not THAT bad)

d.Black = d.Black.map(unidecode)
d.Black = d.Black.map(unidecode)

# first just filter out all weirdness in the dataset, 
# and make a clean name field

d = d[(d.Black.map(lambda x: len(x.split(','))) == 2) & (d.White.map(lambda x: len(x.split(','))) == 2)]
players = list(set(d.White).union(set(d.Black)))
players.sort()
players = pd.DataFrame(players, columns=['player'])

players[['FirstName', 'LastName']] = pd.DataFrame.from_records(list(players.player.map(lambda name: {'FirstName': name.split(',')[1].strip().split(' ')[0].strip('.'), 'LastName': name.split(',')[0].strip().strip('. ')})))
players = players[players.FirstName.map(len) >=3]
players = players[players.LastName.map(len)  >=3]
players.FirstName = players.FirstName.map(lambda n: ''.join(list(filter(lambda x: x.isalpha(), n))))
players.LastName = players.LastName.map(lambda n: ''.join(list(filter(lambda x: x.isalpha(), n))))
players['CleanName'] = players.apply(lambda row: row.LastName + ', ' + row.FirstName, axis=1)
players.to_csv('../data/csv/caissa_players.csv')

players_replacement_dict = {row.player:row.CleanName for _, row in players.iterrows()}

d.Black = d.Black.map(lambda x: players_replacement_dict.get(x,np.nan))
d.White = d.White.map(lambda x: players_replacement_dict.get(x,np.nan))

d.dropna(subset=['White','Black'], inplace=True)

d.reset_index(inplace=True, drop=True)

# it remains to deal with players having the same initial
# in this case i leave only two letters of the first name
# unfortunately, Artyom and Arkady would be the same
# but except for these cases, it's not that bad

players = list(set(d.White).union(set(d.Black)))
players.sort()
players = pd.DataFrame(players, columns=['player'])
players[['FirstName', 'LastName']] = pd.DataFrame.from_records(list(players.player.map(lambda name: {'FirstName': name.split(',')[1].strip(), 'LastName': name.split(',')[0].strip()})))
players['Initial'] = players.FirstName.map(lambda n: n[0])

def resolve_names_same_initial(df):
    if df.shape[0] == 1:
        df.FirstName = df.FirstName.map(lambda n: n[0])
        return df
    if (df.FirstName.map(len) == 1).any():
        df.FirstName = df.FirstName.map(lambda n: n[:min(len(n),1)])
    else:
        df.FirstName = df.FirstName.map(lambda n: n[:2])
    return df

players = players.groupby(['LastName','Initial']).apply(resolve_names_same_initial)

players['CleanName'] = players.apply(lambda row: (row.LastName + ', ' + row.FirstName).lower(), axis=1)

players_replacement_dict = {row.player:row.CleanName for _, row in players.iterrows()}

d.Black = d.Black.map(lambda x: players_replacement_dict.get(x,np.nan))
d.White = d.White.map(lambda x: players_replacement_dict.get(x,np.nan))

d.dropna(subset=['White','Black'], inplace=True)

d = d[(d.BlackElo < 2900) & (d.WhiteElo < 2900)].reset_index(drop=True)

f = lambda pgn: sum([len(m.strip().split()) for m in split(r'\d+\.', pgn) if len(m) > 0])

d.PlyCount = d.pgn.map(f)

d = d.sort_values('Year').reset_index(drop=True)
d.to_csv('../data/csv/caissa_clean.csv')

# now we need to filter out players who played less than 500 games for individual-based comparisons

players = list(set(d.White).union(set(d.Black)))
players.sort()
players = pd.DataFrame(players, columns=['Player'])

white_counts = dict(d.White.value_counts())
black_counts = dict(d.Black.value_counts())

players['GameCounts'] = players.Player.map(lambda name: white_counts.get(name,0) + black_counts.get(name,0))
players = players.sort_values(by='GameCounts', ascending=False)

remaining_players = list(players[players.GameCounts > 500].Player)

d.loc[d.White.isin(remaining_players) & d.Black.isin(remaining_players)]\
    .reset_index(drop=True)\
        .to_csv('../data/csv/caissa_clean_only_frequent_players.csv')
