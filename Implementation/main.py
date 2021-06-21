# %%
from optimization import *
from simulation import *
import os.path
import sys
from matplotlib import pyplot as plt

# Read Data
my_path = os.path.dirname(__file__)
input_data = read_data(os.path.join(my_path, 'Data', 'Data_1d.xlsx'))
# input_data.model_param.gamma = 0.9999
# %% Optimization - Single Policy

# Phase 1
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]
feasible_list = generate_feasible_sa_list(input_data, state_action_list)

# Phase 2
optimal_list, betas = generate_optimal_sa_list(input_data, feasible_list)
export_betas(betas, os.path.join(my_path, 'Data', f'Optimal-Betas-nv_1d_dc.xlsx'))

# %% Simulation - Single Policy

betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-nv_1d_dc.xlsx'))
fig, axes = plt.subplots(1, 1)
results = compare_policies(input_data, betas, 5, 2000, 1000, axes)

print(results)
plt.show()
fig.savefig('sim_res_single.pdf')  

# %% Optimization - Sensitivity Analysis
cw_array = [1, 1.25, 1.5]
cs_array = [1, 1.25, 1.5]
cc_array = [1, 50, 500]
 
# Optimization 
for cw_p in cw_array:
    for cs_p in cs_array:
        for cc_p in cc_array:

            # Read Data
            input_data.model_param.cw = cw_p
            input_data.model_param.cs = cs_p
            input_data.model_param.cc = cc_p

            # Phase 1
            init_state, init_action = generate_initial_state_action(input_data)
            state_action_list = [(init_state, init_action)]
            feasible_list = generate_feasible_sa_list(input_data, state_action_list)

            # Phase 2
            optimal_list, betas = generate_optimal_sa_list(input_data, feasible_list)
            export_betas(betas, os.path.join(my_path, 'Data', f'Optimal-Betas-cw{cw_p}-cs{cs_p}-cc{cc_p}.xlsx'))

# Simulation
tot_scenarios = len(cw_array) * len(cs_array) * len(cc_array)
fig, axes = plt.subplots(1, tot_scenarios)
fig.set_size_inches(10, tot_scenarios*7)
fig.subplots_adjust(top = 1.5)

counter = 0
for cw_p in cw_array:
    for cs_p in cs_array:
        for cc_p in cc_array:
            
            # Read Data
            input_data.model_param.cw = cw_p
            input_data.model_param.cs = cs_p
            input_data.model_param.cc = cc_p
            betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-cw{cw_p}-cs{cs_p}-cc{cc_p}.xlsx'))

            # Compare Policies
            discounted_costs = compare_policies(input_data, betas, 5, 2000, 1000, axes)

            # Plotting
            axes.set_title(f"COSTS: cw - {cw_p}, cs - {cs_p}, cc - {cc_p}, Myopic Cost - {discounted_costs['myopic']}, MDC Cost - {discounted_costs['mdp']}")
            counter += 1
plt.savefig('sim_res_sensitivity.pdf')  

# %% Test out a policy
betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-nv_1d_dc.xlsx'))

# test_out_policy(input_data, 20, myopic_policy, "Myopic")
test_out_policy(input_data, 20, mdp_policy, "MDP", betas)
# %%
