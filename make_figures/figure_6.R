library(tidyverse)
library(scales)

make_beta_plot <- function(data, title) {
  p <- ggplot(data) + 
    geom_segment(aes(x=ll, xend = hh, y=variable, yend=variable), linewidth=0.7) +
    geom_segment(aes(x=l, xend = h, y=variable, yend=variable), linewidth=1.2) + 
    geom_point(aes(x=m, y=variable), size=2) + 
    geom_vline(xintercept=0, color="gray", alpha=0.4) +
    facet_wrap(vars(response), ncol=1) +
    xlab("Coefficient") + ylab("Value") + ggtitle(title) +
    scale_y_discrete(labels=label_parse()) +
    theme_bw()
  return(p)
}

qp_fitness_data <- read.csv("../model/model_results/queens_pawn_ply_2/mcmc_intervals_fitness.csv")
qp_beta_data <- read.csv("../model/model_results/queens_pawn_ply_2/mcmc_intervals_beta.csv")

ck_fitness_data <- read.csv("../model/model_results/carokann_ply_5/mcmc_intervals_fitness.csv")
ck_beta_data <- read.csv("../model/model_results/carokann_ply_5/mcmc_intervals_beta.csv")

naj_fitness_data <- read.csv("../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_fitness.csv")
naj_beta_data <- read.csv("../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_beta.csv")

qp_responses <- qp_fitness_data$response[1:7]
ck_responses <- ck_fitness_data$response[1:6]
naj_responses <- naj_fitness_data$response[1:10]

qp_beta_data$response <- qp_responses
ck_beta_data$response <- ck_responses
naj_beta_data$response <- naj_responses

qp_beta_data$variable <- c(rep("beta[win]",7),rep("beta[top50-win]",7),rep("beta[top50-freq]",7))
ck_beta_data$variable <- c(rep("beta[win]",6),rep("beta[top50-win]",6),rep("beta[top50-freq]",6))
naj_beta_data$variable <- c(rep("beta[win]",10),rep("beta[top50-win]",10),rep("beta[top50-freq]",10))

qp_b <- make_beta_plot(qp_beta_data, "Queen's Pawn, ply 2")
ggsave("../figures/figure_6a.pdf", qp_b,w=2.5,h=5.5,unit="in")

ck_b <- make_beta_plot(ck_beta_data, "Caro-Kann, ply 5")+xlim(-0.3,0.3)
ggsave("../figures/figure_6b.pdf", ck_b,w=2.5,h=5,unit="in")

naj_b <- make_beta_plot(naj_beta_data, "Najdorf Sicilian, ply 11") 
ggsave("../figures/figure_6c.pdf",naj_b,w=2.5,h=8,unit="in")