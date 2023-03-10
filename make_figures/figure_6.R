library(tidyverse)
library(scales)
library(ggh4x)

make_beta_plot <- function(data, title) {
  p <- ggplot(data) +
    geom_segment(aes(x = ll, xend = hh, y = variable, yend = variable), linewidth = 0.7) +
    geom_segment(aes(x = l, xend = h, y = variable, yend = variable), linewidth = 1.2) +
    geom_point(aes(x = m, y = variable), size = 2) +
    geom_vline(xintercept = 0, color = "gray", alpha = 0.4) +
    facet_wrap2(vars(response), nrow = 10, ncol = 1, trim_blank = FALSE) +
    xlab("Coefficient") +
    ylab("Value") +
    ggtitle(title) +
    scale_y_discrete(labels = label_parse()) +
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
      axis.title = element_text(size = 15),
      axis.title.y = element_text(margin = margin(0, 5, 0, 0)),
      strip.text = element_text(size = 12)
    )
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

qp_beta_data$variable <- c(rep("beta[win]", 7), rep("beta[top50-win]", 7), rep("beta[top50-freq]", 7))
ck_beta_data$variable <- c(rep("beta[win]", 6), rep("beta[top50-win]", 6), rep("beta[top50-freq]", 6))
naj_beta_data$variable <- c(rep("beta[win]", 10), rep("beta[top50-win]", 10), rep("beta[top50-freq]", 10))

qp_b <- make_beta_plot(qp_beta_data, "Queen's Pawn, ply 2")
ggsave("../figures/figure_6a.pdf", qp_b, w = 3.3, h = 9, unit = "in")
ggsave("../figures/figure_6a.png", qp_b, w = 3.3, h = 9, unit = "in", dpi = 500)

ck_b <- make_beta_plot(ck_beta_data, "Caro-Kann, ply 5") + xlim(-0.3, 0.3)
ggsave("../figures/figure_6b.pdf", ck_b, w = 3.3, h = 9, unit = "in")
ggsave("../figures/figure_6b.png", ck_b, w = 3.3, h = 9, unit = "in", dpi = 500)

naj_b <- make_beta_plot(naj_beta_data, "Najdorf Sicilian, ply 11")
ggsave("../figures/figure_6c.pdf", naj_b, w = 3.3, h = 9, unit = "in")
ggsave("../figures/figure_6c.png", naj_b, w = 3.3, h = 9, unit = "in", dpi = 500)
