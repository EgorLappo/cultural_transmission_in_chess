library(tidyverse)
library(scales)
library(patchwork)

make_fitness_curve_plot <- function(data, title) {
  fitness_plot_grid <- ggplot(data, aes(xmin=lower_bp, xmax=upper_bp)) +
    geom_rect(aes(ymin=ll,ymax=hh), fill="cadetblue1", alpha = 0.4) +
    geom_rect(aes(ymin=l,ymax=h), fill="cadetblue4", alpha=0.8) +
    geom_line(aes(x=mid_bp,y=m)) +
    facet_grid(rows=vars(response)) + #, scales = "free_y") + 
    scale_x_continuous(trans="log", limits = c(0.0001,1), breaks=c(0.01, 0.1, 0.25,0.5,1)) + 
    theme_bw() + theme(axis.text.x = element_text(angle = 45)) +
    xlab("Move frequency") + ylab(expression(f[i])) + ggtitle(title)
  return(fitness_plot_grid)
}

qp_fitness_data <- read.csv("../model/model_results/queens_pawn_ply_2/mcmc_intervals_fitness.csv")
ck_fitness_data <- read.csv("../model/model_results/carokann_ply_5/mcmc_intervals_fitness.csv")
naj_fitness_data <- read.csv("../model/model_results/sicilian_najdorf_ply_11/mcmc_intervals_fitness.csv")

qp_fitness_data$strategy <- "Queen's Pawn, ply 2"
ck_fitness_data$strategy <- "Caro-Kann, ply 5"
naj_fitness_data$strategy <- "Najdorf sicilian, ply 11"

qp_f <- make_fitness_curve_plot(qp_fitness_data, "Queen's Pawn, ply 2")
ggsave("../figures/figure_5a.pdf", qp_f,w=3,h=5.5,unit="in")

ck_f <- make_fitness_curve_plot(ck_fitness_data, "Caro-Kann, ply 5")
ggsave("../figures/figure_5b.pdf",ck_f,w=3,h=5,unit="in")

naj_f <- make_fitness_curve_plot(naj_fitness_data,"Najdorf Sicilian, ply 11") 
ggsave("../figures/figure_5c.pdf",naj_f,w=3,h=8,unit="in")

