# Cultural transmission of move choice in chess
## Model and figure code

To reproduce the figures in the paper, follow these steps:

0. Clone this repository to your machine.

1. Download the dataset from [Caissabase](http://caissabase.co.uk), unzip it.

2. Use [Scid vs. PC](https://scidvspc.sourceforge.net) to export all games into a PGN file. Save the result as `data/pgn/caissa.pgn` (use [this guide](https://chess.stackexchange.com/questions/13116/how-do-you-convert-a-scid-database-into-a-pgn-database)). Exported `.pgn` is available at **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)** in case you don't want to install weird chess software just to replicate someone's work.

3. Run `parse.py`, `clean.py`, and `generate_tables.py` from the `data_processing/` folder. The required packages can be installed with `pip3 install -r requirements.txt`. 
The parsing takes **a long time** (~11 hrs), since it parses all games to verify that the pgn is coded correctly. I have made **cleaned `.csv` files** available for download at **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)**. If you use these files, you only need to run `generate_tables.py`.

4. Fit the model by running `model/prepare_data.py` and `run_stan.R`. You will need to install `cmdstanr` following [this guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html). You will also need to install other required R packages: `c("tidyverse", "bayesplot", "posterior", "scales", "ggh4x")`.

5. Run each of the scripts in the `make_figures/` folder to generate figures.

6. Email me at `elappo@stanford.edu` with any questions or issues.

## Nix users

This directory is a Nix flake, so you can use `nix develop` to get the environment to reproduce my analysis. Just download the data and run the scripts! This currently only works for `x86` systems due to a pre-compiled header issue with the `cmdstan` package, so most probably this won't work on Apple Silicon Macs.
