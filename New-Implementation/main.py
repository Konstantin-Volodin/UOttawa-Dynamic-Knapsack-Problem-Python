import optimization_new
import simulation_new
import time

# Test Type
test_modifier = "cw1-cc1-cv10-gam95-"
data_type = "simple"

# Optimization Paramers
iter_lims = 100000
beta_fun = [
    (0,0.75),
    (1000,0.95)
    # (10,0.8),
    # (100,0.9),
    # (500,0.95),
    # (10000,0.99)
]
sub_mip_gap = 0.001
import_data_opt = f"sens-data/{data_type}/{test_modifier}{data_type}-data.xlsx"
export_data_opt = f"sens-data/{data_type}/betas/{test_modifier}{data_type}-optimal.pkl"

# Simulation Parameters
replications = 3
warm_up = 1000
duration = 3000
show_policy = False
import_data_sim = import_data_opt
import_betas_sim = export_data_opt
export_txt_sim = f"sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.txt"
export_pic_sim = f"sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.html"

# Execute
start_time_opt = time.time()
optimization_new.main_func(iter_lims, beta_fun, sub_mip_gap, import_data_opt, export_data_opt)
end_time_opt = time.time()
simulation_new.main_func(replications, warm_up, duration, show_policy, import_data_sim, import_betas_sim, export_txt_sim, export_pic_sim)
end_time_sim = time.time()
print(f'{test_modifier}{data_type}\tOptimization: {end_time_opt-start_time_opt} sec \tSimulation: {end_time_sim-end_time_opt} sec')