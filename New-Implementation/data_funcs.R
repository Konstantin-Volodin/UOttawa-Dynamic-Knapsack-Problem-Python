library(tidyverse)
library(here)

warm <- 250
dur <- 1250 
repl <- 30
path <- "R1R2R3"

generate_summary <- function(path, dur, warm, repl) {
  
  # Data
  data <- read_data(path)
  
  # Generate Wait Time
  avg_pw <- analyse_wait(data, dur, warm, repl)
  # Generate Wait List
  avg_wtl <- analyse_wait_list(data, dur, warm, repl)
  # Generate Utilization
  avg_util <- analyse_util(data, dur, warm, repl)
  
gf}



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
analyse_wait <- function(data, dur, warm, repl) {
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

  # Average wait time
  pw_avg <- pwu_arr_nc %>%
    filter(period > warm) %>% group_by(policy) %>%
    summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
    mutate(w_moe = 1.96 * w_sd / sqrt(repl)) %>%
    mutate(surgery = 'overall')
  
  # Surgery wait time
  pw_srg <- pwu_arr_nc %>% 
    filter(period > warm) %>% group_by(policy, surgery) %>%
    summarize(w_m = mean(wait), w_sd = sd(wait)) %>%
    mutate(w_moe = 1.96 * w_sd / sqrt(repl))  
  
  # All data
  pw <- bind_rows(pw_avg, pw_srg) %>% 
    select(policy, surgery, w_m, w_moe) %>%
    mutate(surgery = case_when(
      surgery == 'overall' ~ "Overall",
      surgery == "1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR" ~ "Surgery1",
      surgery == "4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR" ~ "Surgery4",
      surgery == "6. SPINE POSTERIOR DISCECTOMY LUMBAR" ~ "Surgery6"
    )) %>%
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
analyse_wait_list <- function(data, dur, warm, repl) {
  log <- data$log
  
  # Post-warmup waitlist
  pwu_wtl <- bind_rows(
    data$log %>% filter(st_ac == 'post-state' & value == 'psp' & t > 1),
    data$log %>% filter(st_ac == 'post-state' & value == 'pwp')
  ) %>% group_by(policy, repl, period, c) %>%
    summarize(wl = sum(val)) %>% ungroup() %>%
    complete(policy, repl, c, period) %>%
    replace_na(list(wl = 0)) %>%
    rename(surgery = c) %>%
    filter(period > warm)

  # Average wait list
  wtl_avg <- pwu_wtl %>% 
    group_by(policy, repl, period) %>%
    summarize(wl_t = sum(wl)) %>%
    group_by(policy) %>%
    summarize(wl_m = mean(wl_t), wl_sd = sd(wl_t)) %>%
    mutate(wl_moe = 1.96 * wl_sd / sqrt(repl)) %>%
    mutate(surgery = 'overall')
  
  # Surgery wait list
  wtl_srg <- pwu_wtl %>% 
    group_by(policy, surgery) %>%
    summarize(wl_m = mean(wl), wl_sd = sd(wl)) %>%
    mutate(wl_moe = 1.96 * wl_sd / sqrt(repl))
  
  wtl <- bind_rows(wtl_avg, wtl_srg) %>% 
    select(policy, surgery, wl_m, wl_moe) %>%
    mutate(surgery = case_when(
      surgery == 'overall' ~ "Overall",
      surgery == "1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR" ~ "Surgery1",
      surgery == "4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR" ~ "Surgery4",
      surgery == "6. SPINE POSTERIOR DISCECTOMY LUMBAR" ~ "Surgery6"
    )) %>%
    mutate(val = paste0(round(wl_m,2), " += ", round(wl_moe,2))) %>%
    select(policy, surgery, val) %>% 
    pivot_wider(names_from = surgery, values_from = val)
  
  return(wtl)
  
}
### Analyses Utilization
analyse_util <- function(data, dur, warm, repl) {
  # Utilization
  util <- data$util
  
  # Average Utilization
  util_m <- util %>%
    filter(period > warm) %>%
    group_by(policy, resource) %>%
    summarize(ut_m = mean(util), ut_sd = sd(util)) %>%
    mutate(ut_moe = 1.96 * ut_sd / sqrt(repl)) %>% 
    mutate(val = paste0(round(ut_m,2), " += ", round(ut_moe,2))) %>%
    select(policy, resource, val) %>%
    pivot_wider(names_from = resource, values_from = val)
  
  return(util_m)
  
  # bx_util_p <- ggplot(util %>% filter(period > warmup)) +
  #   geom_boxplot(aes(y=util, x=resource, fill=policy)) +
  #   theme_minimal()
  # util %>% filter(period > warmup) %>%
  #   group_by(policy, resource) %>%
  #   summarize(util_m = mean(util), util_sd = sd(util)) %>%
  #   mutate(util_me = 1.96 * util_sd / sqrt(30))
}
### Analysis of reschedules
analyse_resch <- function(data, dur, warm, repl) {
  state <- data$state
  
  # Post Warm Up Reschedules
  pwu_rsc <- state %>% 
    filter(action == 'rescheduled')  %>%
    group_by(policy,surgery,repl,period) %>%
    summarize(resch=n()) %>% 
    ungroup() %>%
    complete(policy, surgery, repl, period=seq(dur)) %>%
    replace_na(list(resch = 0)) %>%
    filter(period > warm)
  
  # Average Reschedules
  rsc_avg <- pwu_rsc %>% 
    group_by(policy, repl, period) %>%
    summarize(rs = sum(resch)) %>%
    group_by(policy) %>%
    summarize(rs_m = mean(rs), rs_sd = sd(rs)) %>%
    mutate(rs_moe = 1.96 * rs_sd / sqrt(repl)) %>%
    mutate(surgery = 'overall')
  
  # Surgery Reschedules
  rsc_srg <- pwu_rsc %>% 
    group_by(policy, surgery) %>%
    summarize(rs_m = mean(resch), rs_sd = sd(resch)) %>%
    mutate(rs_moe = 1.96 * rs_sd / sqrt(repl))
  
  rsc <- bind_rows(rsc_avg, rsc_srg) %>% 
    select(policy, surgery, rs_m, rs_moe) %>%
    mutate(surgery = case_when(
      surgery == 'overall' ~ "Overall",
      surgery == "1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR" ~ "Surgery1",
      surgery == "4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR" ~ "Surgery4",
      surgery == "6. SPINE POSTERIOR DISCECTOMY LUMBAR" ~ "Surgery6"
    )) %>%
    mutate(val = paste0(round(rs_m,2), " += ", round(rs_moe,2))) %>%
    select(policy, surgery, val) %>% 
    pivot_wider(names_from = surgery, values_from = val)
  
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
  
  
}
### Analysis of transitions
analyse_trans <- function(data, dur, warm, repl) {
  
}