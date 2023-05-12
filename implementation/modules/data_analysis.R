library(tidyverse)
library(readr)
library(gridExtra)
library(here)
library(scales)
library(gridExtra)
library(plotly)
library(reticulate)

# LOAD FUNCS
source(here('modules','data_funcs.R'))

# PARAMS
warm <- 250
dur <- 1000
repl <- 30
path <- here('data','full-sm')


# GENERATE DATA
modif <- '0-1'
dt_pl <- generate_summary(path, modif, dur, warm, repl)
dt_sa <- generate_summary_sa(path, modif, dur, warm, repl)
dt_zs <- generate_summary_zs(path, modif, FALSE)
  

