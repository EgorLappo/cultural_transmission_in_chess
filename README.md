# Cultural transmission of move choice in chess

Code accompanying the manuscript "Cultural transmission of move choice in chess" by Lappo, Rosenberg, and Feldman (2023). Preprint available at [arxiv.org/abs/2302.10375](https://arxiv.org/abs/2302.10375).

## Reproducing analysis with Nix (preferred)

First, please ensure that you are working on a machine with an Intel or AMD processor (i.e. `x86_64` architecture). (If you are working on the new Mac computer, follow the guide below that does not use Nix. This is because the `cmdstan` package has issues with selecting the right precompiled headers.)

**0.** [Install nix](https://zero-to-nix.com/concepts/nix-installer) and clone this repository to your machine. To do this, first run
```
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```
to install nix.

Make sure your machine has [`git`](https://git-scm.com/downloads) and [`git-lfs`](https://git-lfs.com) installed, and then run any of the following commands to clone the repository.
```
gh repo clone EgorLappo/cultural_transmission_in_chess
git clone https://github.com/EgorLappo/cultural_transmission_in_chess.git
```

Finally, run `git lfs fetch` to make sure the data file is downloaded.

**1.** Run `nix develop` to log into a shell with all necessary programs and packages.

**2.** Decompress the data file by running

```
cd data/csv
ouch decompress caissa_clean.tar.gz
```

This file represents the exact dataset we used. If you want to access the original database of games, it is available from [Caissabase](http://caissabase.co.uk). You will need to use [Scid vs. PC](https://scidvspc.sourceforge.net) to export all games into a PGN file and put it into `data/pgn` (use [this guide](https://chess.stackexchange.com/questions/13116/how-do-you-convert-a-scid-database-into-a-pgn-database)). To then produce the `.csv` file, run `parse.py` and `clean.py` from the `data_processing` folder. You do not need to run `parse.py` and `clean.py` if you use the `.csv` file we provide. Note that [Caissabase](http://caissabase.co.uk) data may be updated in the future by adding more games.

All the `.csv` files we use in our analysis are also duplicated at **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)**.

**3.** Run the script to generate additional tables.

```
cd data_processing
python generate_tables.py
```

**4.** Fit the model.

```
cd model
python prepare_data.py
Rscript --vanilla run_stan.R
```

**5.** Run each of the scripts in the `make_figures/` folder to generate figures.

```
cd make_figures
python figure_2.py
python figure_3.py
python figure_4.py
python figure_5.py
python figure_s2.py
python figure_s3.py
python figure_s4.py
python figure_s5.py
Rscript --vanilla figure_s1.R
```

**6.** Email me at `elappo@stanford.edu` with any questions or issues.

## Reproducing analysis without Nix

To reproduce analysis without the help of the Nix system, clone the repository and decompress the source `.csv` file as described above. Then, make sure that your version of python has all the necessary packages installed by running `pip3 install -r requirements.txt`. Creating an environment with [`miniforge`](https://github.com/conda-forge/miniforge) can help. Install `cmdstanr` following [this guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html). Finally, install required R packages: `c("tidyverse", "bayesplot", "posterior", "scales", "ggh4x")`.

Now you are ready to run all the scripts. Follow the guide above starting with 3.