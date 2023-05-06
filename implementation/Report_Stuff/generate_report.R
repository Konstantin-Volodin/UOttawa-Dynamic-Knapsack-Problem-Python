library(reticulate)
reticulate::use_condaenv('uOttawa-dyn-knap-R', required=TRUE)
source_python("z_factor.py")

R1R2_beta_file <- "Data/sens-data/smaller-full/betas/cw1-cc5-cv10-gam99-smaller-full-nopri-optimal-R1R2.pkl"
R1R2R3_beta_file <- 
pickle_data <- return_plots(R1R2_beta_file, FALSE)