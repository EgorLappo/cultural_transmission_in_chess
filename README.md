# Cultural transmission of move choice in chess
## Model and figure code

To reproduce the figures in the paper, follow these steps:

0. Clone this repository to your machine.

1. Download the dataset from [Caissabase](http://caissabase.co.uk), unzip it.

2. Use [Scid vs. PC](https://scidvspc.sourceforge.net) to export all games into a PGN file. Save the result as `data/pgn/caissa.pgn`. (Use [this guide](https://chess.stackexchange.com/questions/13116/how-do-you-convert-a-scid-database-into-a-pgn-database)) Exported `.pgn` is available at **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)** if you don't want to install weird chess software just to replicate someone's work.

3. Run `parse_and_clean.py` and `generate_tables.py` from the `data_processing/` folder. The required packages are listed in `requirements.txt` file (you can install them with `pip3 install -r requirements.txt`). 
This takes **a long time** (~11 hrs), since it parses all games to verify that the pgn is coded correctly. So, I have provided **clean `.csv` files** available for download if you trust my data cleaning and processing steps. Follow **[this GDrive link](https://drive.google.com/drive/folders/1rBVvs7kwwfCKchg5htEdzNUZlxkBiRO7?usp=sharing)** to download the clean tables. If you use the downloaded files, you only need to run `generate_tables.py`.

4. Fit the model by running `model/prepare_data.py` and `run_stan.R`. You will need to install `cmdstanr` following [this guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html). You will also need to install other required R packages: `c("tidyverse", "bayesplot", "posterior", "scales")`.

5. Run each of the scripts in the `make_figures/` folder to generate all figures.

6. Email me at `elappo@stanford.edu` with any questions or issues.
