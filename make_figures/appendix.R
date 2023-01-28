library(tidyverse)
library(scales)

make_fitness_curve_plot <- function(data, title) {
  fitness_plot_grid <- ggplot(data, aes(xmin=lower_bp, xmax=upper_bp)) +
    geom_rect(aes(ymin=ll,ymax=hh), fill="cadetblue1", alpha = 0.4) +
    geom_rect(aes(ymin=l,ymax=h), fill="cadetblue3", alpha=0.8) +
    geom_line(aes(x=mid_bp,y=m)) +
    facet_grid(rows=vars(response)) + #, scales = "free_y") + 
    scale_x_continuous(trans="log", limits = c(0.0001,1), breaks=c(0.01, 0.1, 0.25,0.5,1)) + 
    theme_bw() + theme(axis.text.x = element_text(angle = 45)) +
    xlab("Move frequency") + ylab("") + ggtitle(title)
  return(fitness_plot_grid)
}

make_beta_plot <- function(data, title) {
  p <- ggplot(data) + 
  geom_segment(aes(x=ll, xend = hh, y=variable, yend=variable), linewidth=0.7) +
  geom_segment(aes(x=l, xend = h, y=variable, yend=variable), linewidth=1.2) + 
  geom_point(aes(x=m, y=variable), size=2) + 
  geom_vline(xintercept=0, color="gray", alpha=0.4) +
  facet_wrap(vars(response), ncol=2) +
  xlab("Coefficient") + ylab("Value") + ggtitle(title) +
  scale_y_discrete(labels=label_parse()) +
  theme_bw()
  return(p)
}

qp_fitness_data <- read.csv("../model/model_results/queens_pawn_ply_2/mcmc_intervals_fitness.csv")
qp_beta_data <- read.csv("../model/model_results/queens_pawn_ply_2/mcmc_intervals_beta.csv")

kp_fitness_data <- read.csv("../model/model_results/kings_pawn_ply_5/mcmc_intervals_fitness.csv")
kp_beta_data <- read.csv("../model/model_results/kings_pawn_ply_5/mcmc_intervals_beta.csv")

naj_fitness_data <- read.csv("../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_fitness.csv")
naj_beta_data <- read.csv("../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_beta.csv")

qp_responses <- qp_fitness_data$response[1:7]
kp_responses <- kp_fitness_data$response[1:5]
naj_responses <- naj_fitness_data$response[1:10]

qp_beta_data <- qp_beta_data[2:22, ]
kp_beta_data <- kp_beta_data[2:16, ]
naj_beta_data <- naj_beta_data[2:31, ]

qp_beta_data$response <- qp_responses
kp_beta_data$response <- kp_responses
naj_beta_data$response <- naj_responses

qp_beta_data$variable <- c(rep("beta[win]",7),rep("beta[top50-win]",7),rep("beta[top50-freq]",7))
kp_beta_data$variable <- c(rep("beta[win]",5),rep("beta[top50-win]",5),rep("beta[top50-freq]",5))
naj_beta_data$variable <- c(rep("beta[win]",10),rep("beta[top50-win]",10),rep("beta[top50-freq]",10))

qp_f <- make_fitness_curve_plot(qp_fitness_data, "Queen's Pawn, ply 2")
ggsave("../figures/appendix/qp_fitness_curves.pdf", qp_f,w=3,h=5,unit="in")

kp_f <- make_fitness_curve_plot(kp_fitness_data, "King's Pawn, ply 5")
ggsave("../figures/appendix/kp_fitness_curves.pdf",kp_f,w=3,h=4,unit="in")

naj_f <- make_fitness_curve_plot(naj_fitness_data,"Najdorf Sicilian, ply 11") 
ggsave("../figures/appendix/najdorf_fitness_curves.pdf",naj_f,w=3,h=8,unit="in")

qp_b <- make_beta_plot(qp_beta_data, "Queen's Pawn, ply 2")
ggsave("../figures/appendix/qp_betas_plot.pdf", qp_b,w=5,h=5,unit="in")

kp_b <- make_beta_plot(kp_beta_data, "King's Pawn, ply 5")
ggsave("../figures/appendix/kp_betas_plot.pdf", kp_b,w=5,h=4,unit="in")

naj_b <- make_beta_plot(naj_beta_data, "Najdorf Sicilian, ply 11") 
ggsave("../figures/appendix/najdorf_betas_plot.pdf",naj_b,w=5,h=6.5,unit="in")