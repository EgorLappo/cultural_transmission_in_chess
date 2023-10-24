library(tidyverse)
library(bayesplot)
library(patchwork)

qp_fit <- readRDS("../model/model_fits/queens_pawn_ply_2/fit.RDS")
ck_fit <- readRDS("../model/model_fits/carokann_ply_5/fit.RDS")
n_fit <- readRDS("../model/model_fits/sicilian_najdorf_ply_11/fit.RDS")

print(qp_fit$diagnostic_summary())
print(ck_fit$diagnostic_summary())
print(n_fit$diagnostic_summary())

qp_rhat <- rhat(qp_fit$clone(), pars = c("beta_win", "beta_win_top", "beta_freq_top", "fitness_values"))
ck_rhat <- rhat(ck_fit, pars = c("beta_win", "beta_win_top", "beta_freq_top", "fitness_values"))
n_rhat <- rhat(n_fit, pars = c("beta_win", "beta_win_top", "beta_freq_top", "fitness_values"))

qp_rhat_plot <- mcmc_rhat_hist(qp_rhat) + ggtitle("Queen's pawn, ply 2")
ck_rhat_plot <- mcmc_rhat_hist(ck_rhat) + ggtitle("Caro-Kann, ply 5")
n_rhat_plot <- mcmc_rhat_hist(n_rhat) + ggtitle("Najdorf Sicilian, ply 11")

rhat_plot <- qp_rhat_plot + ck_rhat_plot + n_rhat_plot + plot_annotation(tag_levels = "A") & theme_classic() & theme(plot.tag = element_text(face = "bold"), text = element_text(family = "Helvetica")) & legend_none()

ggsave("../figures/figure_s1.pdf", rhat_plot, height = 3, width = 8, units = "in")
