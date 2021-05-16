# Title     : Analysis of RT
# Objective : Test effect of cue-probe incongrunecy
# Created by: Jose C. Garcia Alanis
# Created on: 23.07.20
# R version : 4.0.2 (2020-06-22), Taking Off Again

# source function for fast loading and installing of packages
source('./r_functions/pkgcheck.R')

# 1) define the path to behavioral data directory -----------------------------

# get system and user information
host <- Sys.info()

# set default path or promt user for other path
if (grep('jose', host['user']) & grep('x', host['sysname'])) {

  # defaut path in project structure
  path_to_rt <- '../data_bids/derivatives/results'

} else {

  path_to_rt <- readline('Please provide path to behavioral data: ')

}

# 2) import in the data -------------------------------------------------------
# this part requires the package 'dplyr'
pkgcheck('dplyr')

rt <- tibble()
for (pattern in c('congruent', 'incongruent', 'mixed')) {

  # files in dir
  rt_files <- list.files(path = paste(path_to_rt, 'rt', sep = '/'),
                         pattern = paste0('*_', pattern, '*'),
                         full.names = T)

  # read in the files
  rt_list <- lapply(rt_files, read.table, sep = '\t', header = T)

  # put the in a data frame
  rt_df <- bind_rows(rt_list, .id = "column_label")

  # recode block variable
  rt_df <- rt_df %>%
    mutate(block = paste(pattern, 'stroop'))

  rt <- rbind(rt, rt_df)

}

# 2) import in the subject info -----------------------------------------------
group <- read.table('../data_bids/subject_data/subject_ids.tsv',
                    header = T,
                    sep = '\t')

mci_ids <- unique(group[group$group_id == 'MCI', 'subject_id'])
control_ids <- unique(group[group$group_id == 'Control', 'subject_id'])

corrects <- rt %>%
  filter(reaction == 'correct') %>%
  mutate(group = ifelse(subject %in% mci_ids, 'MCI', 'Control')) %>%
  group_by(subject, condition, block, group) %>%
  summarise(rt = mean(rt))


# 2) import in the data -------------------------------------------------------
# this part requires the package 'dplyr'
pkgcheck(c('ggplot2', 'ggforce', 'Hmisc'))

pd <- position_dodge(0.35)
rt_plot <- ggplot(data = corrects,
                  aes(x = condition,
                      y = rt,
                      color = group,
                      fill = group, shape = group)) +
  ggforce::facet_row(facets = vars(block),
                      scales = 'free_x',
                      space = 'free') +
  stat_summary(fun = mean, geom = "point", position = pd) +
  stat_summary(fun.data = mean_se,
               geom = "errorbar",
               position = pd, width = 0.35) +
  scale_y_continuous(breaks = seq(0.80, 1.80, by=0.2)) +
  coord_cartesian(ylim = c(0.80, 1.80)) +
  scale_shape_manual(values = c(25, 24)) +
  geom_segment(aes(x = -Inf, y = 0.80, xend = -Inf, yend = 1.80),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = -Inf, y = -Inf,
                   xend = Inf, yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 10),
        panel.background = element_rect(fill = 'white'),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                colour = "gray97"),
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                colour = "gray97"),
        strip.text = element_text(color = 'black', size = 12),
        strip.background = element_blank(),
        legend.position='bottom',
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        panel.spacing = unit(1, "lines")); rt_plot
# save to disk
ggsave(filename = '../data_bids/derivatives/results/rt.pdf',
       plot = rt_plot, width = 7, height = 5, dpi = 300)



#
# compute number of trials per condition
total <- rt %>%
  group_by(subject, condition, block) %>%
  mutate(n_trials = sum(!is.na(trial))) %>%
  select(subject, condition, block, n_trials) %>%
  arrange(subject, condition, block) %>%
  unique()

# compute number of errors per condition
errors <- rt %>%
  filter(reaction == 'incorrect') %>%
  group_by(subject, condition, block) %>%
  mutate(n_errors = sum(!is.na(trial))) %>%
  summarise(n_errors = mean(n_errors)) %>%
  arrange(subject, condition, block)

# merge data frames
errors <- merge(total, errors, c('subject', 'condition', 'block'), all.x = T)
# replace missing values with zeros
errors[is.na(errors)] <- 0

errors <- errors %>%
  mutate(error_rate=(n_errors+0.5)/(n_trials+1))

n_errors <- errors %>%
  mutate(group = ifelse(subject %in% mci_ids, 'MCI', 'Control'))


pd <- position_dodge(0.35)
errors_plot <- ggplot(data = n_errors,
                  aes(x = condition,
                      y = error_rate,
                      color = group,
                      fill = group, shape = group)) +
  ggforce::facet_row(facets = vars(block),
                      scales = 'free_x',
                      space = 'free') +
  stat_summary(fun = mean, geom = "point", position = pd) +
  stat_summary(fun.data = mean_se,
               geom = "errorbar",
               position = pd, width = 0.35) +
  scale_y_continuous(breaks = seq(0.0, 0.5, by = 0.1)) +
  coord_cartesian(ylim = c(0.0, 0.5)) +
  scale_shape_manual(values = c(25, 24)) +
  geom_segment(aes(x = -Inf, y = 0.0, xend = -Inf, yend = 0.5),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = -Inf, y = -Inf,
                   xend = Inf, yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 10),
        panel.background = element_rect(fill = 'white'),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                colour = "gray97"),
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                colour = "gray97"),
        strip.text = element_text(color = 'black', size = 12),
        strip.background = element_blank(),
        legend.position='bottom',
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        panel.spacing = unit(1, "lines")); errors_plot
# save to disk
ggsave(filename = '../data_bids/derivatives/results/errors.pdf',
       plot = errors_plot, width = 7, height = 5, dpi = 300)
