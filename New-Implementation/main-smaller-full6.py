import optimization_new
import simulation_new    
import time

# Test Type
test_modifier = "cw1-cc5-cv1000-gam95-"
data_type = "smaller-full"

# Optimization Paramers
iter_lims = 1000000
beta_fun = [
    (0,0.75),
    (10000,0.8)
]
sub_mip_gap = 0.001
import_data_opt = f"Data/sens-data/{data_type}/{test_modifier}{data_type}-data.xlsx"
export_data_opt = f"Data/sens-data/{data_type}/betas/{test_modifier}{data_type}-optimal.pkl"
export_data_p2 = f"Data/sens-data/{data_type}/full-model/{test_modifier}{data_type}-p2.mps"
import_data_p2 = export_data_p2

# Simulation Parameters
replications = 3
warm_up = 2000
duration = 6000
show_policy = False
import_data_sim = import_data_opt
import_betas_sim = export_data_opt
export_txt_sim = f"Data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.txt"
export_pic_sim = f"Data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.html"
export_state_my = f"Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-state-my.txt"
export_state_md = f"Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-state-md.txt"

# Execute
start_time_opt = time.time()
optimization_new.main_func(iter_lims, beta_fun, sub_mip_gap, import_data_opt, export_data_opt, export_data_p2, import_data_p2)
end_time_opt = time.time()
simulation_new.main_func(replications, warm_up, duration, show_policy, import_data_sim, import_betas_sim, export_txt_sim, export_pic_sim, export_state_my, export_state_md)
end_time_sim = time.time()
print(f'{test_modifier}{data_type}\tOptimization: {end_time_opt-start_time_opt} sec \tSimulation: {end_time_sim-end_time_opt} sec')
print(f'{beta_fun}')