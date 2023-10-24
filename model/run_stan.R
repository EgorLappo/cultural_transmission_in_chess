library(tidyverse)
library(cmdstanr)
library(posterior)
library(bayesplot)

env <- Sys.getenv()

if (!is.na(env["NIX"])) {
  set_cmdstan_path(env["CMDSTANPATH"])
}

run_stan_model <- function(strategy_name, pieces) {
  # load data
  count_data <- read.csv(paste0(
    "data/",
    strategy_name, "/response_counts.csv"
  ))
  win_data <- read.csv(paste0(
    "data/",
    strategy_name, "/win_rates.csv"
  ))
  top_player_count_data <- read.csv(paste0(
    "data/",
    strategy_name, "/top_player_response_counts.csv"
  ))
  top_player_win_data <- read.csv(paste0(
    "data/",
    strategy_name, "/top_player_win_rates.csv"
  ))

  # make matrices
  count_matrix <- as.matrix(count_data[, 2:ncol(count_data)]) 

  win_matrix <- as.matrix(win_data[, 2:ncol(win_data)])
  win_matrix <- scale(win_matrix)
  win_matrix[is.na(win_matrix)] <- 0.0

  top_player_count_matrix <- as.matrix(
    top_player_count_data[, 2:ncol(top_player_count_data)]
  )

  top_player_win_matrix <- as.matrix(
    top_player_win_data[, 2:ncol(top_player_win_data)]
  )
  top_player_win_matrix <- scale(top_player_win_matrix)
  top_player_win_matrix[is.na(top_player_win_matrix)] <- 0.0

  # compute frequencies from counts
  top_player_freq_matrix <- top_player_count_matrix /
    rowSums(top_player_count_matrix)
  top_player_freq_matrix <- scale(top_player_freq_matrix)
  top_player_freq_matrix[is.na(top_player_freq_matrix)] <- 0.0

  freq_matrix <- count_matrix / rowSums(count_matrix)

  # freq_quantiles_by_strategy <- apply(freq_matrix, 2,
  #        quantile, probs = c(0.25, 0.5, 0.75))
  freq_quantiles_by_strategy <- apply(
    freq_matrix, 2,
    function(x) quantile(x[x != 0], probs = c(0.25, 0.5, 0.75))
  )
  freq_quantiles_by_strategy <- t(rbind(0, freq_quantiles_by_strategy, 1))

  if (pieces == "black") {
    win_matrix <- 0.0-win_matrix
    top_player_win_matrix <- 0.0-top_player_win_matrix
  }

  data_list <- list(
    N = nrow(count_matrix),
    K = ncol(count_matrix),
    response_counts = count_matrix,
    top_player_response_frequencies = top_player_freq_matrix,
    response_frequencies = freq_matrix,
    win_rates = win_matrix,
    top_player_win_rates = top_player_win_matrix,
    alpha_prior = rep(1, ncol(count_matrix)),
    R = 4,
    breakpoints = freq_quantiles_by_strategy
  )

  # compile model
  model <- cmdstan_model("model.stan")

  fit <- model$sample(
    data = data_list,
    seed = 123,
    chains = 4,
    parallel_chains = 4,
    refresh = 500, # print update every 500 iters
    iter_warmup = 2000,
    iter_sampling = 10000
  )

  dir.create("model_fits/")
  dir.create(paste0("model_fits/", strategy_name))

  fit$save_object(file = paste0("model_fits/", strategy_name, "/fit.RDS"))
  write.csv(
    freq_quantiles_by_strategy,
    paste0("model_fits/", strategy_name, "/frequency_quantiles_by_strategy.csv")
  )
}


summarize_fits <- function(strategy_name) {
  # create a directory for all models assiociated to the strategy
  dir.create("model_results")
  dir.create(paste0("model_results/", strategy_name))

  # read the original data to get the moves that we have been analyzing
  responses <- paste0("data/", strategy_name, "/response_counts.csv") %>%
    read.csv() %>%
    colnames() %>%
    tail(-1)
  n <- length(responses)

  fit <- readRDS(paste0("model_fits/", strategy_name, "/fit.RDS"))

  fit %>%
    as_draws_df() %>%
    write.csv(paste0("model_fits/", strategy_name, "/fit.csv"))

  # the only thing that changes here is that breakpoints are different for each response
  breakpoints_strat <- read.csv(paste0("model_fits/", strategy_name, "/frequency_quantiles_by_strategy.csv"))
  breakpoints_strat <- as.matrix(breakpoints_strat)[, 2:ncol(breakpoints_strat)] %>% apply(2, as.numeric)
  breakpoints_strat[, 1] <- 0.0001
  r <- ncol(breakpoints_strat) - 1

  fitness_draws <- fit$draws(variables = "fitness_values") %>%
    as_draws_df() %>%
    mcmc_intervals_data(prob_outer = 0.98)
  # "spread out" (from matrix to vector) the things we have to this big dataframe
  fitness_draws$response <- rep(responses, r)
  fitness_draws$lower_bp <- as.vector(breakpoints_strat)[1:(n * r)]
  fitness_draws$upper_bp <- as.vector(breakpoints_strat)[(n + 1):(n * (r + 1))]
  fitness_draws$mid_bp <- fitness_draws$upper_bp / 2 + fitness_draws$lower_bp / 2

  write_csv(fitness_draws, paste0("model_results/", strategy_name, "/mcmc_intervals_fitness.csv"))
  fit$draws(variables = c("beta_win", "beta_win_top", "beta_freq_top")) %>%
    as_draws_df() %>%
    mcmc_intervals_data(prob_outer = 0.98) %>%
    write_csv(paste0("model_results/", strategy_name, "/mcmc_intervals_beta.csv"))
}


strategy_names <- c(
  "queens_pawn_ply_2",
  "carokann_ply_5",
  "sicilian_najdorf_ply_11"
)

pieces <- c("black", "white", "white")

for (i in 1:length(strategy_names)) {
  run_stan_model(strategy_names[i], pieces[i])
  summarize_fits(strategy_names[i])
}
