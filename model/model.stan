functions {
    real piecewise_constant(real x, int R, vector values, vector breakpoints) {
        for (r in 1:R) {
            if(x > breakpoints[r] && x <= breakpoints[r+1]) {
                return(values[r]);
            }
        }
        return(0.0);
    }
}

data {
  int<lower=1> N; // number of observations/years
  int<lower=1> K; // number of responses
  array[N,K] int response_counts; // counts of each strategy
  array[N,K] real response_frequencies; // frequencies of each strategy
  array[N,K] real top_player_response_frequencies;
  array[N] vector[K] win_rates; // win rates of each strategy
  array[N] vector[K] top_player_win_rates;
  int<lower=1> R; // number of breakpoints for fitness function
  array[K] vector[R+1] breakpoints; // breakpoints for fitness function
  vector[K] alpha_prior; // prior for alpha
}

parameters {
  vector[K] beta_win;
  vector[K] beta_win_top;
  vector[K] beta_freq_top;
  array[K] vector<lower=0>[R] fitness_values;
  array[N-1] simplex[K] p;
}

model {
  for (j in 1:K) {
    beta_win[j] ~ normal(0,1);
    beta_win_top[j] ~ normal(0,1);
    beta_freq_top[j] ~ normal(0,1);
    for (r in 1:R) {
      fitness_values[j, r] ~ exponential(1);
    }
  }



  array[N-1] vector[K] a;

  for(i in 1:(N-1)) {

    for (k in 1:K) {
      a[i,k] = alpha_prior[k]
             + piecewise_constant(response_frequencies[i, k], R, fitness_values[k], breakpoints[k])
             * exp(beta_win[k] * win_rates[i,k]
             + beta_win_top[k] * top_player_win_rates[i,k]
             + beta_freq_top[k] * top_player_response_frequencies[i,k])
             * response_counts[i,k];
    }

    p[i] ~ dirichlet(a[i]);

    response_counts[i+1] ~ multinomial(p[i]); // counts in the next generation
  }
}

generated quantities {
  array[N-1] vector[K] fitness;
  array[N-1] vector[K] meanprobs;
  array[N-1] real ll;
  real log_lik;

  for (i in 1:(N-1)) {
    vector[K] a;
    real asum;

    for (k in 1:K) {
      fitness[i,k] = piecewise_constant(response_frequencies[i, k], R, fitness_values[k], breakpoints[k]);
      a[k] = alpha_prior[k]
             + fitness[i,k]
             * exp(beta_win[k] * win_rates[i,k]
             + beta_win_top[k] * top_player_win_rates[i,k]
             + beta_freq_top[k] * top_player_response_frequencies[i,k])
             * response_counts[i,k];
    }

    asum = sum(a);

    for (k in 1:K) {
      meanprobs[i,k] = a[k]/asum;
    }

    ll[i] = multinomial_lpmf(response_counts[i+1] | p[i]);
  }
  log_lik = sum(ll);
}
