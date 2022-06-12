library(tidyverse)
library(readr)
library(gridExtra)

warmup <- 1000
duration <- 3000

### READ DATA
# State Data
state_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-state-md-less-arr-.txt")
state_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-state-my-less-arr-.txt")
state <- bind_rows(state_md, state_my)
rm(state_md,state_my)
# Cost Data
cost_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-cost-md-less-arr-.txt") %>% mutate(policy = "MDP")
cost_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-cost-my-less-arr-.txt")  %>% mutate(policy = "Myopic")
cost <- bind_rows(cost_md, cost_my)
rm(cost_md,cost_my)
# Utilization Data
util_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-util-md-less-arr-.txt") %>% mutate(policy = "MDP") %>% filter(horizon_period == 0)
util_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-util-my-less-arr-.txt") %>% mutate(policy = "Myopic") %>% filter(horizon_period == 0)
util <- bind_rows(util_md, util_my) %>% 
  mutate(bed = usage_admin/1.5, OR = usage_OR/11.25) %>%
  select(repl, period, policy, bed, OR) %>%
  pivot_longer(cols = c('bed','OR'), names_to='resource', values_to = 'util')
rm(util_md,util_my)


### COST ANALYSIS
# Timeseries Cost
ts_cost <- cost %>% group_by(period, policy) %>% summarise(cost_m = mean(cost), cost_sd = sd(cost))
ts_cost_p <- ggplot(ts_cost, aes(x=period)) +
  geom_line(aes(y=cost_m, color=policy)) +
  geom_ribbon(aes(ymin=cost_m- (2*cost_sd/sqrt(30)), ymax=cost_m+ (2*cost_sd/sqrt(30)), fill=policy), alpha=0.3) +
  theme_minimal() + 
  theme(legend.position="bottom")
bx_cost_p <- ggplot(cost %>% filter(period > warmup)) +
  geom_boxplot(aes(y=cost, fill=policy)) +
  theme_minimal() + 
  theme(legend.position="bottom")



### UTILIZATION ANALYSIS
# Utilization
bx_util_p <- ggplot(util %>% filter(period > warmup)) +
  geom_boxplot(aes(y=util, x=resource, fill=policy)) +
  theme_minimal()


### WAIT TIME ANALYSIS
# Post warm-up Patients
pwu_pat <- state %>% 
  group_by(policy, repl, id) %>% 
  filter(action %in% c('arrived')) %>%
  ungroup() %>%
  filter(arrived_on > warmup) %>%
  select(policy, repl, id)
# Post warm-up wait time (only completed)
pwu_wait_c <- state %>%
  group_by(policy, repl, id) %>% 
  filter(action %in% c('arrived','scheduled', 'rescheduled')) %>% 
  slice(c(1,n())) %>%
  mutate(final_sched = case_when(
    is.na(sched_to) == F ~ sched_to,
    is.na(resch_to) == F ~ resch_to
  )) %>%
  mutate(final_sched = min(final_sched, na.rm=T)) %>%
  slice(c(1)) %>%
  filter(final_sched != Inf) %>%
  select(policy, repl, id, period, priority, complexity, surgery, arrived_on, final_sched) %>%
  ungroup() %>%
  mutate(wait = final_sched - arrived_on)
# Post warm-up wait time (including not completed)
pwu_wait_nc <- state %>%
  group_by(policy, repl, id) %>% 
  filter(action %in% c('arrived','scheduled', 'rescheduled')) %>% 
  slice(c(1,n())) %>%
  mutate(final_sched = case_when(
    is.na(sched_to) == F ~ sched_to,
    is.na(resch_to) == F ~ resch_to,
    TRUE ~ duration+1
  )) %>%
  mutate(final_sched = min(final_sched, na.rm=T)) %>%
  slice(c(1)) %>%
  select(policy, repl, id, period, priority, complexity, surgery, arrived_on, final_sched) %>%
  ungroup() %>%
  mutate(wait = final_sched - arrived_on)
#graphs
bx_wait_c <- ggplot(pwu_wait_c %>% right_join(pwu_pat) %>% drop_na()) +
  geom_boxplot(aes(y=wait, x=surgery, fill=policy)) +
  facet_grid(complexity ~ priority)
bx_wait_nc <- ggplot(pwu_wait_nc %>% right_join(pwu_pat)) +
  geom_boxplot(aes(y=wait, x=surgery, fill=policy)) +
  facet_grid(complexity ~ priority)



### WAITLIST SIZE
#data
periods <- pwu_wait_nc %>% distinct(period) %>% arrange(period) %>% pull(period)
wl_data <- tibble()
for (p in periods) {
  print(p)
  wait_data <- pwu_wait_nc %>%
    group_by(policy, repl, .drop=FALSE) %>%
    filter(arrived_on <= p & final_sched > p) %>% 
    summarize(wt_size = n()) %>%
    group_by(policy) %>%
    summarize(wt_size_m = mean(wt_size), wt_size_sd = sd(wt_size)) %>%
    mutate(period = p)
  wl_data <- bind_rows(wl_data, wait_data)
}

#graphs
ts_wl_p <- ggplot(wl_data, aes(x=period)) +
  geom_line(aes(y=wt_size_m , color=policy)) +
  geom_ribbon(aes(ymin=wt_size_m - (2*wt_size_sd / sqrt(30)), ymax=wt_size_m + (2*wt_size_sd / sqrt(30) ), fill=policy), alpha=0.3) +
  theme_minimal() + 
  theme(legend.position="bottom") + 
  labs(x = 'Week #', y='Size of the Waitlist', title='Waitlist Size over time')
bx_wl_p <- ggplot(wl_data %>% filter(period > warmup)) +
  geom_boxplot(aes(y=wt_size_m , fill=policy)) +
  theme_minimal() + 
  theme(legend.position="bottom") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  labs(y='Size of the Waitlist', title='Waitlist Boxplot')



### Reschedules Data  
# Usage of Reschedules
resch_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'rescheduled' & period > warmup) %>%
  group_by(repl,policy) %>% 
  summarize(resch = n()) %>%
  ungroup()
bx_resch <- ggplot(resch_data, aes(x=policy, y=resch, fill=policy)) +
  geom_boxplot() + 
  theme_minimal()




### FINAL PLOTS
# Cost
grid.arrange(ts_cost_p, bx_cost_p, ncol=2)
# utilization
bx_util_p
# wait time
bx_wait_c
bx_wait_nc
# wait list size
grid.arrange(ts_wl_p, bx_wl_p, ncol=2)
# reschedules
bx_resch
