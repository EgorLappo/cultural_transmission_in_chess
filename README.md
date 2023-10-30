# Cultural transmission of move choice in chess

Code accompanying the manuscript "Cultural transmission of move choice in chess" by Lappo, Rosenberg, and Feldman (2023), available at [doi.org/10.1098/rspb.2023.1634](https://doi.org/10.1098/rspb.2023.1634). 

## Reproducing model fitting and figures with Nix (preferred)

First, please ensure that you are working on a machine with an Intel or AMD processor (i.e. `x86_64` architecture). (If you are working on a Mac computer with an M1/M2 processor, follow the guide below that does not use Nix. This is because the `cmdstan` package has issues with selecting the right precompiled headers.)

**0.** [Install nix](https://zero-to-nix.com/concepts/nix-installer) and download the code repository with data. To do this, first run
```
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```
to install nix.

Then, download the repository that includes the input data file from Zenodo: [zenodo.org/doi/10.5281/zenodo.10038192](https://zenodo.org/doi/10.5281/zenodo.10038192).

**1.** Run `nix develop` to log into a shell with all scripts and packages necessary for processing data, fitting the model of move choice, and generating all figures. **Note:** this may take a long time.

**2.** The data file `data/csv/caissa_clean.csv` is the exact dataset we used to fit the model and draw figures, containing 3448853 games. If you want to access the original database of games, it is available from [Caissabase](http://caissabase.co.uk). You will need to use [Scid vs. PC](https://scidvspc.sourceforge.net) to export all games into a PGN file and put it into `data/pgn` (use [this guide](https://chess.stackexchange.com/questions/13116/how-do-you-convert-a-scid-database-into-a-pgn-database)). Then, run scripts `parse.py` and `clean.py` from the `data_processing` folder to produce the `.csv` table `caissa_clean.csv`. You do not need to run `parse.py` and `clean.py` if you use the `.csv` file we provide on Zenodo. Note that [Caissabase](http://caissabase.co.uk) data may be updated in the future by retroactively adding more games, in which case the results can be slightly different.

All the `.csv` files we use in our analysis are also duplicated at **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)**.

**3.** Run the script to generate additional tables `caissa_counts_by_ply_{i}.csv` that will be used in making streamplots in Figure 3.

```
cd data_processing
python generate_tables.py
```

**4.** Fit the Bayesian model of move choice in chess with Stan (Section 5(b)(iv) of the paper).

```
cd model
python prepare_data.py
Rscript --vanilla run_stan.R
```

**5.** Run each of the scripts in the `make_figures/` folder to generate all figures in the main text and the supplement.

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
