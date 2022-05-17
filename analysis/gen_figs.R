library(dplyr)
library(forcats)
library(ggplot2)
library(ggpubr)
library(latex2exp)

setwd('/mnt/data/projects/hdpgmm-music-experiments')
d <- read.csv('./results/results_agg.csv') %>%
  mutate(dataset = fct_recode(dataset,
                              "MTAT" = "magnatagatune",
                              "GTZAN" = "gtzan",
                              "Echonest" = "echonest"))


# PER REGULARIZATION
# 1. raw
d %>% filter(!is.na(regularization) & n_train == '200k') %>%
  ggplot(aes(x=regularization, y=score, group=regularization)) +
  # geom_jitter(width=0.2) +
  geom_boxplot() +
  geom_jitter(alpha=.5) +
  scale_x_continuous(trans='log10') +
  facet_wrap(.~dataset, scales='free_y')

# 2. summary
d.summary <- d %>%
  filter(!is.na(regularization) & n_train == '200k') %>%
  group_by(regularization, dataset) %>%
  summarise(
    sd = sd(score, na.rm =T),
    score = mean(score)
  )
(p <- d %>%
    filter(!is.na(regularization) & n_train == '200k') %>%
    ggplot(aes(x=regularization, y=score)) +
    geom_jitter(position = position_jitter(0.2), alpha = .2) +
    geom_line(aes(group=1, ymin=score - sd, ymax=score + sd), data=d.summary) +
    geom_errorbar(aes(ymin=score - sd, ymax=score + sd), data=d.summary, width = 0.2) +
    geom_point(aes(ymin=score - sd, ymax=score + sd), data=d.summary, size = 2) +
    scale_x_continuous(trans='log10') +
    facet_wrap(.~dataset,
               scales='free_y') +
    theme_pubr() +
    theme(axis.text.x = element_text(angle = 45, hjust=1)))


datasets = list("Echonest", "GTZAN", "MTAT")
acc.measures = list("nDCG@500", "F1[macro]", "AUROC[macro]")
ps = list()
for (i in 1:length(datasets)) {
  d.summary.dataset = d.summary %>% filter(dataset == datasets[[i]])
  
  p_ <- d %>%
    filter(!is.na(regularization) & n_train == '200k' & dataset == datasets[[i]]) %>%
    ggplot(aes(x=regularization, y=score)) +
    geom_jitter(position = position_jitter(0.2), alpha = .2) +
    geom_line(aes(group=1, ymin=score - sd, ymax=score + sd), data=d.summary.dataset) +
    geom_errorbar(aes(ymin=score - sd, ymax=score + sd), data=d.summary.dataset, width = 0.2) +
    geom_point(aes(ymin=score - sd, ymax=score + sd), data=d.summary.dataset, size = 2) +
    scale_x_continuous(trans='log10') +
    xlab(TeX(r'($\eta_{0}$)')) + ylab(acc.measures[[i]]) + facet_wrap(dataset~.) +
    theme_pubclean() +
    theme(axis.text.x = element_text(angle = 45, hjust=1))
  ps[[i]] = p_
}
(p <- ggarrange(plotlist=ps, nrow = 1, ncol = 3))

ggsave('./paper/ismir_submission/figs/regularization_effect.pdf',
       plot=p, width = 2000, height = 900, units="px",
       dpi = 320)



# PER MODEL
d %>% filter((regularization == 1e-1 & n_train == '200k') | is.na(regularization)) %>%
  mutate(model = factor(model,
                        levels=c("g1",
                                 "vqcodebook32",
                                 "vqcodebook64",
                                 "vqcodebook128",
                                 "vqcodebook256",
                                 "kim",
                                 "clmr",
                                 "clmr256",
                                 "hdpgmm"))) %>%
  ggplot(aes(x=model, y=score)) +
  geom_boxplot() +
  facet_wrap(.~dataset, scales='free_y')

d.all.summary <- d %>%
  filter((regularization == 1e-1 & n_train == '200k') | is.na(regularization)) %>%
  group_by(dataset, model) %>%
  summarise(
    sd = sd(score, na.rm =T),
    score = mean(score)
  )


# PER DATASET SIZE
d %>% filter(!is.na(n_train) & regularization == 1e-1) %>%
  mutate(model = factor(model, levels=c("hdpgmm"))) %>%
  ggplot(aes(x=n_train, y=score)) +
  geom_boxplot() +
  facet_wrap(.~dataset, scales='free_y')


d.summary <- d %>%
  filter(!is.na(n_train) & regularization == 1e-1) %>%
  group_by(n_train, dataset) %>%
  summarise(
    sd = sd(score, na.rm =T),
    score = mean(score)
  )
ps = list()
for (i in 1:length(datasets)) {
  d.summary.dataset = d.summary %>% filter(dataset == datasets[[i]])
  
  p_ <- d %>% filter(!is.na(n_train) & regularization == 1e-1 & dataset == datasets[[i]]) %>%
    mutate(n_train = factor(n_train, levels=c("2k", "20k", "200k"))) %>%
    ggplot(aes(x=n_train, y=score)) +
    geom_jitter(position = position_jitter(0.2), alpha = .2) +
    geom_line(aes(group=1, ymin=score - sd, ymax=score + sd), data=d.summary.dataset) +
    geom_errorbar(aes(ymin=score - sd, ymax=score + sd), data=d.summary.dataset, width = 0.2) +
    geom_point(aes(ymin=score - sd, ymax=score + sd), data=d.summary.dataset, size = 2) +
    xlab(TeX(r'($\eta_{0}$)')) + ylab(acc.measures[[i]]) + facet_wrap(dataset~.) +
    theme_pubclean() +
    theme(axis.text.x = element_text(angle = 45, hjust=1))
  ps[[i]] = p_
}
(p <- ggarrange(plotlist=ps, nrow = 1, ncol = 3))
