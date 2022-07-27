library(tidyverse)
library(readr)
library(gridExtra)

warmup <- 250
duration <- 1250

### READ DATA
# State Data
state_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-state-md-nopri-R1R2.txt",
                     col_types = cols(id = col_double(), repl=col_double(), period=col_double(), arrived_on=col_double(), sched_to=col_double(), resch_from=col_double(),
                                      resch_to=col_double())) %>% filter(!is.na(id))
state_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-state-my-nopri-R1R2.txt",
                     col_types = cols(id = col_double(), repl=col_double(), period=col_double(), arrived_on=col_double(), sched_to=col_double(), resch_from=col_double(), 
                                      resch_to=col_double())) %>% filter(!is.na(id))
state <- bind_rows(state_md, state_my)
rm(state_md,state_my)
# Cost Data
cost_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-cost-md-nopri-R1R2.txt") %>% mutate(policy = "MDP")
cost_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-cost-my-nopri-R1R2.txt")  %>% mutate(policy = "Myopic")
cost <- bind_rows(cost_md, cost_my)
rm(cost_md,cost_my)
# Utilization Data
util_md <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-util-md-nopri-R1R2.txt") %>% mutate(policy = "MDP") %>% filter(horizon_period == 0)
util_my <- read_csv("Data/sens-res/smaller-full/state-action/cw1-cc5-cv10-gam99-smaller-full-util-my-nopri-R1R2.txt") %>% mutate(policy = "Myopic") %>% filter(horizon_period == 0)
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
util %>% filter(period > warmup) %>%
  group_by(policy, resource) %>%
  summarize(util_m = mean(util), util_sd = sd(util)) %>%
  mutate(util_me = 1.96 * util_sd / sqrt(30))

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
# Percentage of Reschedules
resch_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'rescheduled' & period > warmup) %>%
  group_by(repl,policy, surgery) %>% 
  summarize(resch = n()) %>%
  ungroup()
arr_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'arrived' & period > warmup) %>%
  group_by(repl,policy, surgery) %>%
  summarize(arrv = n()) %>%
  ungroup()
resch_data %>% left_join(arr_data) %>% mutate(resch_perc = resch/arrv * 100) %>%
  group_by(policy, surgery) %>%
  summarize(rs_m = mean(resch_perc), rs_sd = sd(resch_perc))

resch_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'rescheduled' & period > warmup) %>%
  group_by(repl,policy) %>% 
  summarize(resch = n()) %>%
  ungroup()
arr_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'arrived' & period > warmup) %>%
  group_by(repl,policy) %>%
  summarize(arrv = n()) %>%
  ungroup()
resch_data %>% left_join(arr_data) %>% mutate(resch_perc = resch/arrv * 100) %>%
  group_by(policy) %>%
  summarize(rs_m = mean(resch_perc), rs_sd = sd(resch_perc))

bx_resch <- ggplot(resch_data, aes(x=policy, y=resch, fill=policy)) +
  geom_boxplot() + 
  theme_minimal()

### Transitions Data
# Percentage of Transitions
transit_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'transition' & period > warmup) %>%
  group_by(repl,policy,surgery) %>% 
  summarize(transit = n()) %>%
  ungroup()
arr_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'arrived' & period > warmup) %>%
  group_by(repl,policy,surgery) %>%
  summarize(arrv = n()) %>%
  ungroup()
transit_data %>% left_join(arr_data) %>% mutate(tr_perc = transit/arrv * 100) %>%
  group_by(policy,surgery) %>%
  summarize(tr_m = mean(tr_perc), tr_sd = sd(tr_perc))

transit_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'transition' & period > warmup) %>%
  group_by(repl,policy) %>% 
  summarize(transit = n()) %>%
  ungroup()
arr_data <- state %>% 
  modify_if(is.character, as.factor) %>%
  filter(action == 'arrived' & period > warmup) %>%
  group_by(repl,policy) %>%
  summarize(arrv = n()) %>%
  ungroup()
transit_data %>% left_join(arr_data) %>% mutate(tr_perc = transit/arrv * 100) %>%
  group_by(policy) %>%
  summarize(tr_m = mean(tr_perc), tr_sd = sd(tr_perc))


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




### Review Logging Data
number <- 5
log_my <- read_csv(paste0("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-my#",number,".txt"))
log_my <- log_my %>% mutate(policy = "myopic") %>% mutate(period = as.character(period))
log_md <- read_csv(paste0("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-md#",number,".txt"))
log_md <- log_md %>% mutate(policy = "MDP") %>% mutate(period = as.character(period))
log <- bind_rows(log_md, log_my)
rm(log_md,log_my)

log %>% filter(period == 0 & `state-aciton` == "action") %>% filter(policy == 'MDP')



log_md1 <- read_csv("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-md-nopri-R1R2.txt")
log_md1 <- log_md1 %>% mutate(policy = "MDP") %>% mutate(period = as.character(period))
log_my1 <- read_csv("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-my-nopri-R1R2.txt")
log_my1 <- log_my1 %>% mutate(policy = "myopic") %>% mutate(period = as.character(period))
log1 <- bind_rows(log_md1, log_my1)

log_md2 <- read_csv("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-md-nopri-R1R2.txt")
log_md2 <- log_md1 %>% mutate(policy = "MDP") %>% mutate(period = as.character(period))
log_my2 <- read_csv("Data/sens-res/smaller-full/logging/cw1-cc5-cv10-gam99-smaller-full-sa-my-nopri-R1R2.txt")
log_my2 <- log_my1 %>% mutate(policy = "myopic") %>% mutate(period = as.character(period))
log2 <- bind_rows(log_md2, log_my2)
