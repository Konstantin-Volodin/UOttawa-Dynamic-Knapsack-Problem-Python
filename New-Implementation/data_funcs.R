library(tidyverse)
library(here)

warm <- 250
dur <- 1250 
path <- "R1R2"

generate_summary <- function(path, dur, warm) {
  
  # Data
  data <- read_data(path)
  
  # Generate Wait Time
  avg_pw <- analyse_wait(data, dur, warm)
  # Generate Wait List
  
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
  )) %>% rename(st_ac = `state-aciton`) %>% mutate(policy = 'MDP')
  log_my <- read_csv(here('simulation_data',paste0('SA-my-',path,'.txt')), col_types = cols(
    period=col_double(),`state-aciton`=col_character(),value=col_character(), 
    t=col_double(),tp=col_double(),m=col_double(),val=col_double()
  )) %>% rename(st_ac = `state-aciton`) %>% mutate(policy = "myopic")
  log <- bind_rows(log_md, log_my)
  
  return(
    list('state' = state, 'cost' = cost, 'util' = util, 'log' = log)
  )
  
}
### Analyses Wait Time
analyse_wait <- function(data, dur, warm) {
  state <- data$state
  
  # Post Warm-up Arrivals
  pwu_arr <- state %>% 
    filter(action == 'arrived') %>%
    filter(arrived_on > warm) %>%
    select(policy, repl, period, id)
  
  # Post warm-up wait time (including not completed)
  pwu_arr_nc <- state %>%
    group_by(policy, repl, id) %>% 
    filter(action %in% c('arrived','scheduled', 'rescheduled')) %>% 
    slice(c(1,n())) %>%
    mutate(final_sched = case_when(
      is.na(sched_to) == F ~ sched_to,
      is.na(resch_to) == F ~ resch_to,
      TRUE ~ dur+1
    )) %>%
    mutate(final_sched = min(final_sched, na.rm=T)) %>%
    slice(c(1)) %>% ungroup() %>%
    select(policy, repl, period, id, priority, complexity, surgery, arrived_on, final_sched) %>%
    mutate(wait = final_sched - arrived_on)

  pw_avg <- pwu_arr_nc %>%
    filter(period > warm) %>% group_by(policy) %>%
    summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
    mutate(w_moe = 1.96 * w_sd / sqrt(30)) %>%
    mutate(surgery = 'overall')
  pw_srg <- pwu_arr_nc %>% 
    filter(period > warm) %>% group_by(policy, surgery) %>%
    summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
    mutate(w_moe = 1.96 * w_sd / sqrt(30))  
  pw <- bind_rows(pw_avg, pw_srg) %>% 
    select(policy, surgery, w_m, w_moe) %>%
    mutate(val = paste0(round(w_m,2), " += ", round(w_moe,2))) %>%
    select(policy, surgery, val) %>% 
    pivot_wider(names_from = surgery, values_from = val)
  
  return(pw)
  #graphs
  # bx_wait_c <- ggplot(pwu_wait_c %>% right_join(pwu_pat) %>% drop_na()) +
  #   geom_boxplot(aes(y=wait, x=surgery, fill=policy)) +
  #   facet_grid(complexity ~ priority)
  # bx_wait_nc <- ggplot(pwu_wait_nc %>% right_join(pwu_pat)) +
  #   geom_boxplot(aes(y=wait, x=surgery, fill=policy)) +
  #   facet_grid(complexity ~ priority)
  # 
  # pwu_wait_nc %>% filter(period > warmup) %>%
  #   group_by(policy, surgery) %>%
  #   summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
  #   mutate(w_me = 1.96 * w_sd / sqrt(30)) %>% select(-c(w_sd))
  # pwu_wait_nc %>% filter(period > warmup) %>%
  #   group_by(policy) %>%
  #   summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
  #   mutate(w_me = 1.96 * w_sd / sqrt(30)) %>% select(-c(w_sd))
  # 
}
### Analyses Wait List
analyse_wait_list <- function(data, dur, warm) {
  log <- data$log
  state <- data$state
  
  wl_data <- tibble()
  for (p in seq(dur)) {
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
  wl_data %>% filter(period > warm) %>%
    group_by(policy) %>% 
    summarize(wt_m = mean(wt_size_m), wt_sd = sd(wt_size_sd))  %>%
    mutate(wt_me = 1.96 * (wt_sd / sqrt(30)) )
  
  bind_rows(
    log %>% filter(st_ac == 'state' & value == 'ps'),
    log %>% filter(st_ac == 'state' & value == 'pw')) %>% 
    arrange(policy, repl, period) %>%
    group_by(policy, repl, period) %>%
    summarize(wl = sum(val)) %>%
    group_by(policy) %>%
    summarize(wl_m = mean(wl))
}
