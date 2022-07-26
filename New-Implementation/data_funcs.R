library(tidyverse)
library(here)

generate_summary <- function(path, dur, warm) {
  
  
  
}



### Reads all relevant data
read_data <- function(path) {
  
  # Patient Transition Data
  state_md <- read_csv(here("simulation_data",paste0('STATE-md-',path,'.txt')), col_types = cols(
    id = col_double(), repl=col_double(), period=col_double(), arrived_on=col_double(), 
    sched_to=col_double(), resch_from=col_double(), resch_to=col_double())) %>% filter(!is.na(id))
  state_my <- read_csv(here("simulation_data",paste0('STATE-my-',path,'.txt')), col_types = cols(
    id = col_double(), repl=col_double(), period=col_double(), arrived_on=col_double(), 
    sched_to=col_double(), resch_from=col_double(), resch_to=col_double())) %>% filter(!is.na(id))
  state <- bind_rows(state_md, state_my)

  # Cost Data
  cost_md <- read_csv(here("simulation_data",paste0('COST-md-',path,'.txt'))) %>% mutate(policy = 'MDP')
  cost_my <- read_csv(here("simulation_data",paste0('COST-my-',path,'.txt'))) %>% mutate(policy = 'myopic')
  cost <- bind_rows(cost_md, cost_my)
  
  # Utilization Data
  util_md <- read_csv(here("simulation_data",paste0('UTIL-md-',path,'.txt'))) %>% mutate(policy = 'MDP') %>% filter(horizon_period == 0)
  util_my <- read_csv(here("simulation_data",paste0('UTIL-my-',path,'.txt'))) %>% mutate(policy = 'myopic') %>% filter(horizon_period == 0)
  util <- bind_rows(util_md, util_my) %>% 
    mutate(bed = usage_admin/1.5, OR = usage_OR/11.25) %>%
    select(repl, period, policy, bed, OR) %>%
    pivot_longer(cols = c('bed','OR'), names_to='resource', values_to = 'util')
  
  # Logging Data
  log_md <- read_csv(here('simulation_data',paste0('SA-md-',path,'.txt')), col_types = cols(
    period=col_double(),`state-aciton`=col_character(),value=col_character(), 
    t=col_double(),tp=col_double(),m=col_double(),val=col_double()
  ))
  log_my
}