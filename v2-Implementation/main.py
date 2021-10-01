# %%
from optimization import *
from simulation import *
import Modules.decorators

import os.path
import sys
from matplotlib import pyplot as plt

# Read Data
# my_path = os.path.dirname(__file__)
my_path = os.getcwd()
# print(my_path)
input_data = read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))
# generate_optimal_sa_list = Modules.decorators.timer(generate_optimal_sa_list)

# %% Optimization
# Phase 1
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]
feasible_list = generate_feasible_sa_list(input_data, state_action_list)
 
# Phase 2
# stabilization_parameter = 0.7
# start_betas = read_betas(os.path.join(my_path, 'Data', f'Intermediate-Betas.xlsx'))
# optimal_list, betas = generate_optimal_sa_list(input_data, feasible_list,stabilization_parameter)
# export_betas(betas, os.path.join(my_path, 'Data', f'Optimal-Betas-full-newcost.xlsx'))

# %% Compare Policies
# Import betas
# fig, axes = plt.subplots(1, 1)
# betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-full-newcost.xlsx'))
# compare_policies(input_data, betas, 3, 10000, 5000, axes)
# fig.savefig(os.path.join(my_path, 'Data', f'Optimal-Betas-full-newcost-long.pdf'))

# # %% Test out policies
# input_data.arrival[('Complexity 1', 'CPU 1')] = 20
# betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-full-newcost.xlsx'))
# test_out_policy(input_data, 10, mdp_policy, "MDP", betas)
# test_out_policy(input_data, 10, myopic_policy, "Myopic")

# %% Init
init_state, init_action = generate_initial_state_action(input_data)
sa_l = [(init_state, init_action)]

p2m, p2v, p2c = generate_master_model(input_data, sa_l)
p1m, p1c = generate_phase1_master_model(input_data, p2m)
p1m.Params.LogToConsole = 0
p1m.optimize()
print(p1m.ObjVal)
p1b = generate_beta_values(input_data, p1c)

p1sm, p1sv = generate_sub_model(input_data, p1b, True)
p1sm.Params.LogToConsole = 0
p1sm.optimize()
n_a = generate_state_action(p1sv)
sa_l.append(n_a)

for i in range(10):
    p2m, p2v, p2c = update_master_model(input_data, p2m, p2v, p2c, n_a, i)
    p1m, p1c = generate_phase1_master_model(input_data, p2m)
    p1m.Params.LogToConsole = 0
    p1m.optimize()
    print(p1m.ObjVal)
    p1b = generate_beta_values(input_data, p1c)

    p1sm, p1sv = update_sub_model(input_data, p1sm, p1sv, p1b, True)
    p1sm.Params.LogToConsole = 0
    p1sm.optimize()
    n_a = generate_state_action(p1sv)
    sa_l.append(n_a)
# %% Betas 3
# p2m, p2v, p2c = update_master_model(input_data, p2m, p2v, p2c, n_a, 1)
# p1m, p1c = generate_phase1_master_model(input_data, p2m)
# p1m.optimize()
# p1b = generate_beta_values(input_data, p1c)
