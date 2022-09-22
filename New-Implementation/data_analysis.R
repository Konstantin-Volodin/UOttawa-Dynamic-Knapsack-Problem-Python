library(tidyverse)
library(readr)
library(gridExtra)
library(here)
library(scales)
library(gridExtra)
library(plotly)

source(here('data_funcs.R'))

warm <- 250
dur <- 1250
repl <- 30

my_dat <- read_csv(here("simulation_data/SA-my-R1R2R3.txt")) %>% mutate(policy = 'my')
md_dat <- read_csv(here("simulation_data/SA-md-R1R2R3.txt")) %>% mutate(policy = 'md')
dat <- my_dat %>% bind_rows(md_dat)

# Scheduling
sched_dat <- dat %>% filter(`state-aciton` == 'action' & value == 'sc') %>%
  filter(period >= warm) %>%
  mutate(t = as.numeric(t)) %>%
  group_by(repl, policy, t, c, d) %>% 
  summarize(count = sum(val) / (dur-warm)) %>% 
  group_by(policy, t,c,d) %>%
  summarize(mean = mean(count), sd = sd(count)) %>%
  mutate(mean_log = log(mean)) %>%
  separate(c, sep=" ", into=c('c'))

ggplot(sched_dat) +
  geom_bar(aes(x=t, y=mean, fill=c), stat='identity') +
  geom_text(aes(x=t, y=mean, label=round(mean,1)), size=2) +
  facet_grid(c+d ~ policy, scales="fixed") + 
  labs(x='Time (Week)', y='Count (Sched)', title='Average Scheduling Numbers')

# Rescheduling
rsc_dat <- dat %>% filter(value == 'rsc') %>%
  filter(period >= warm) %>%
  group_by(policy, repl, c, d,t ,tp) %>%
  summarize(avg = sum(val)) %>%
  group_by(policy, c, d, t, tp) %>%
  summarize(mean = mean(avg), sd = sd(avg)) %>%
  separate(c, sep=' ', into=c('c')) %>%
  mutate(t = as.numeric(t), tp = as.numeric(tp))

(ggplot(rsc_dat %>% filter(t == 1)) +
  geom_bar(aes(x=t, y=mean, fill=tp), stat = 'identity', position='dodge2') +
  facet_grid(c+d ~ policy, scales='free') + 
  labs(x='Time (Week)', y='Count (Sched)', title='Good Reschedules per Group') +
  theme_minimal()) %>% ggplotly()

ggplot(rsc_dat %>% filter(t != 1)) +
  geom_bar(aes(x=t, y=mean, fill=tp), stat = 'identity', position='dodge2') +
  facet_grid(c+d ~ policy, scales='free') + 
  labs(x='Time (Week)', y='Count (Sched)', title='Bad Reschedules per Group') +
  scale_color_brewer(palette = 'RdBu') +
  theme_minimal()



# Waitlist
waitlist_dat <- dat %>% filter(value == 'pw') %>%
  filter(period >= warm) %>%
  group_by(policy, repl, c, d, m) %>% 
  summarize(avg = sum(val)/(dur-warm)) %>% 
  group_by(policy, c,d, m) %>%
  summarize(mean = mean(avg), sd = sd(avg)) %>%
  separate(c, sep=" ", into=c('c'))

ggplot(waitlist_dat) +
  geom_bar(aes(x=m, y=mean, fill=c), stat = 'identity') +
  geom_text(aes(x=m, y=mean, label=round(mean,1)), size=2) +
  facet_grid(c+d ~ policy) + 
  labs(x='Time (Week)', y='Count (Sched)', title='Average Waitlist Size by Group')


waitlist_dat <- dat %>% filter(value == 'pw') %>%
  filter(period >= warm) %>%
  group_by(policy, repl, c, d, m) %>% 
  summarize(avg = sum(val)/(dur-warm)) %>% 
  group_by(policy, c,d, m) %>%
  summarize(mean = mean(avg), sd = sd(avg)) %>%
  separate(c, sep=" ", into=c('c'))

ggplot(waitlist_dat) +
  geom_bar(aes(x=m, y=mean, fill=c), stat = 'identity') +
  geom_text(aes(x=m, y=mean, label=round(mean,1)), size=2) +
  facet_grid(c+d ~ policy) + 
  labs(x='Time (Week)', y='Count (Sched)', title='Average Waitlist Size by Group')








# warm <- 250
# dur <- 1250 
# repl <- 30
# path <- "R1R2R3"
# R1R2R3 <- generate_summary(path, dur, warm, repl)
# 
# warm <- 250
# dur <- 1250 
# repl <- 30
# path <- "R1R2"
# R1R2 <- generate_summary(path, dur, warm, repl)
# 
# 
# r1r2r3_st <- R1R2R3$data$state %>% mutate(type = 'R1R2R3')
# r1r2_st <- R1R2R3$data$state %>% mutate(type = 'R1R2')
# 
# st <- bind_rows(r1r2r3_st, r1r2_st)
