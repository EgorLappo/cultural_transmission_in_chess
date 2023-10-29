import pandas as pd
from re import split
from tqdm import tqdm

## COUNT GAMES BY INITIAL PLYS

d = pd.read_csv('../data/csv/caissa_clean.csv', index_col=0)

f = lambda pgn: sum([m.strip().split() for m in split(r'\d+\.', pgn) if len(m) > 0],[])

def extract_first_plys(row):
    plys = {}
    plys['Year'] = row.Year
    plys['Result'] = row.Result
    pgn = f(row.pgn)
    for i in range(16):
        if i < len(pgn):
            plys['Ply'+str(i+1)] = pgn[i]
        else:
            plys['Ply'+str(i+1)] = pd.NA
    
    return plys

ply_rows = []

for _, row in tqdm(d.iterrows()):
    ply_rows.append(extract_first_plys(row))

d_plys = pd.DataFrame.from_records(ply_rows)
d_plys = d_plys[['Year']+list(d_plys.columns[2:])+['Result']]

for i in tqdm(range(12)):
    d_plys_gb = d_plys.groupby(list(d_plys.columns[:i+2])).agg({d_plys.columns[i+2]: 'count', 'Result': 'mean'}).reset_index()
    d_plys_gb.columns = list(d_plys_gb.columns[:-2]) + ['Count', 'Result']
    d_plys_gb['Moves'] = d_plys_gb[d_plys_gb.columns[1:i+2]].agg(' '.join, axis=1)
    d_plys_gb.to_csv('../data/csv/caissa_counts_by_ply'+str(i+1)+'.csv')