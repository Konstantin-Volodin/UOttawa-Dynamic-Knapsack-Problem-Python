library(tidyverse)
library(readr)
library(gridExtra)
library(here)

source(here('data_funcs.R'))

warm <- 250
dur <- 1250 
repl <- 30
path <- "R1R2R3"
R1R2R3 <- generate_summary(path, dur, warm, repl)

warm <- 250
dur <- 1250 
repl <- 30
path <- "R1R2"
R1R2 <- generate_summary(path, dur, warm, repl)


r1r2r3_st <- R1R2R3$data$state %>% mutate(type = 'R1R2R3')
r1r2_st <- R1R2R3$data$state %>% mutate(type = 'R1R2')

st <- bind_rows(r1r2r3_st, r1r2_st)
