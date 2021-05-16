# %% Imports
from Modules.data_import import *
from Modules.data_export import *
from Modules.simulation import *
import os.path

my_path = os.path.dirname(__file__)
input_data = read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))
betas = read_betas(os.path.join(my_path, 'Data', 'Optimal-Betas.xlsx'))

# %% Checking Logic
def non_zero_state(state: state):
    for key,value in state.ue_tp.items():
        if state.ue_tp[key] >= 0.1: print(f'\tUnits Expected - {key} - {state.ue_tp[key]}')
        if state.uu_tp[key] >= 0.1: print(f'\tUnits Used - {key} - {state.uu_tp[key]}')
    for key,value in state.pw_mdc.items(): 
        if value >= 0.1: print(f'\tPatients Waiting- {key} - {value}')
    for key,value in state.ps_tmdc.items(): 
        if value >= 0.1: print(f'\tPatients Scheduled- {key} - {value}')
def non_zero_action(action: action):
    for key,value in action.sc_tmdc.items():
        if value >= 0.1: print(f'\tPatients Schedule - {key} - {value}')
    for key,value in action.rsc_ttpmdc.items(): 
        if value >= 0.1: print(f'\tPatients Reschedule- {key} - {value}')
    for key,value in action.uv_tp.items(): 
        if value >= 0.1: print(f'\tUnits Violated- {key} - {value}')
    for key,value in action.uu_p_tp.items(): 
        if value >= 0.1: print(f'\tUnits Used - Post Decision - {key} - {value}')
    for key,value in action.pw_p_mdc.items(): 
        if value >= 0.1: print(f'\ttPatients Waiting - Post Decision - {key} - {value}')
    for key,value in action.ps_p_tmdc.items(): 
        if value >= 0.1: print(f'\tPatients Scheduled - Post Decision - {key} - {value}')

# %% Expected Values Simulation
cost_data = generate_expected_values(input_data, 30, 1000)
fas_costs = simulation(input_data, 10, 600, 300, fas_policy)
myopic_cost = simulation(input_data, 10, 600, 300, myopic_policy)
fas_cost_total = 0
myopic_cost_total = 0
for repl in fas_costs:
    for day in repl:
        fas_cost_total += day
for repl in myopic_cost:
    for day in repl:
        myopic_cost_total += day
print(fas_cost_total)
print(myopic_cost_total)

# %% Comparison of Policies
# Myopic Policy
myopic_cost = simulation(input_data, 10, 500, 100, myopic_policy, False)
myopic_cost_total = 0
for repl in myopic_cost:
    for day in repl:
        myopic_cost_total += day

# MDP Policy
mdp_cost = simulation(input_data, 10, 500, 100, mdp_policy, False, betas=betas)
mdp_cost_total = 0
for repl in mdp_cost:
    for day in repl:
        mdp_cost_total += day
# %% Writing Out Things
curr_state = initial_state(input_data)
curr_action = initial_action(input_data)

for i in range(5):
    print()
    print(f'Day {i}: Initial State')
    non_zero_state(curr_state)

    print('MDP Action')
    new_action = mdp_policy(input_data, curr_state, betas)
    non_zero_action(new_action)

    print(f'Post Action State')
    curr_state = execute_action(input_data, curr_state, new_action)
    non_zero_state(curr_state)
    # print(f'\t Cost: {state_action_cost(input_data, curr_state, new_action)}')

    curr_state = execute_transition(input_data, curr_state)