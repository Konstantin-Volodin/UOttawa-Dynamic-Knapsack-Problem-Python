# %%
from optimization import *
from simulation import *
import Modules.decorators
import os.path
import sys
from matplotlib import pyplot as plt

# Read Data
my_path = os.path.dirname(__file__)
input_data = read_data(os.path.join(my_path, 'Data', 'Data-full.xlsx'))
generate_optimal_sa_list = Modules.decorators.timer(generate_optimal_sa_list)
# %% Optimization
# Phase 1
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]
feasible_list = generate_feasible_sa_list(input_data, state_action_list)

# Phase 2
stabilization_parameter = 0.3
acceptable_error = 0
optimal_list, betas = generate_optimal_sa_list(input_data, feasible_list,stabilization_parameter, acceptable_error)
export_betas(betas, os.path.join(my_path, 'Data', f'Optimal-Betas-full.xlsx'))
            


# # %% Compare Policies
# # Import betas
# fig, axes = plt.subplots(1, 1)
# betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-full.xlsx'))
# compare_policies(input_data, betas, 5, 3000, 1500, axes)
# fig.show()

# # %% Test out policies
# betas = read_betas(os.path.join(my_path, 'Data', f'Optimal-Betas-full.xlsx'))
# # input_data.arrival[('Complexity 1', 'CPU 1')] = 30
# test_out_policy(input_data, 10, mdp_policy, "MDP", betas)
# test_out_policy(input_data, 100, myopic_policy, "Myopic")

# %%
