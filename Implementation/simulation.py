# %% Imports
from Modules.data_import import *
from Modules.data_export import *
from Modules.simulation import *
import os.path
import pandas as pd
import numpy as np
import seaborn as sns


# %% Expected Values Simulation
def expected_val_sim(input_data, path):
    expected_values = generate_expected_values(input_data, 5, 1000, 500)
    export_expected_vals(os.path.join(path, 'Data', 'Data.xlsx'), expected_values)

# %% Comparison of Policies
def compare_policies(input_data, betas, repls, days, warm_up, axes):
    repls = repls
    total_days = days
    warm_up = warm_up

    # Myopic Policy (FAS)
    np.random.seed(10)
    myopic_cost, myopic_disc_c, myopic_state = simulation(input_data, repls, total_days, warm_up, myopic_policy)
    # MDP Policy
    np.random.seed(10)
    mdp_cost, mdp_disc_c, mdp_state = simulation(input_data, repls, total_days, warm_up, mdp_policy, betas=betas)
    
    # Visualization
    myopic_cost_avg = pd.DataFrame(myopic_cost).transpose().mean(axis=1)
    myopic_cost_avg = pd.DataFrame(myopic_cost_avg, columns=['Myopic'])
    mdp_cost_avg = pd.DataFrame(mdp_cost).transpose().mean(axis=1)
    mdp_cost_avg = pd.DataFrame(mdp_cost_avg, columns=['MDP'])
    cost_frame = pd.concat([myopic_cost_avg, mdp_cost_avg], axis=1)
    cost_frame['Day'] = cost_frame.index

    sns.lineplot(data=cost_frame, x="Day", y="Myopic", label='Myopic', ax=axes)
    sns.lineplot(data=cost_frame, x="Day", y="MDP", label='MDP', ax=axes)

    print(f"Myopic Discounted Cost - {np.average(myopic_disc_c)}")
    print(f"MDP Discounted Cost - {np.average(mdp_disc_c)}")

    res = {'myopic': np.average(myopic_disc_c), 'mdp': np.average(mdp_disc_c)}
    return res

# %% Writing Out Things
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
def test_out_policy(input_data, duration, policy, policy_name, betas = False):

    curr_state = initial_state(input_data)

    for i in range(duration):
        print(f'Day {i}: Initial State')
        non_zero_state(curr_state)

        print(f'{policy_name} Action')
        new_action = None
        if betas:
            new_action = policy(input_data, curr_state, betas)
        else:
            new_action = policy(input_data, curr_state)
        non_zero_action(new_action)

        print(f'Post Action State')
        curr_state = execute_action(input_data, curr_state, new_action)
        non_zero_state(curr_state)
        print(f'\t Cost: {state_action_cost(input_data, curr_state, new_action)}')

        curr_state = execute_transition(input_data, curr_state, new_action)
