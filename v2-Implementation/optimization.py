# %% Packages
from Modules.data_import import *
from Modules.data_export import *
from Modules.master_model import *
from Modules.sub_problem import * 
import gurobipy as gp

import Modules.decorators
# generate_master_model = Modules.decorators.timer(generate_master_model)
# generate_phase1_master_model = Modules.decorators.timer(generate_phase1_master_model)
# update_master_model = Modules.decorators.timer(update_master_model)
# generate_sub_model = Modules.decorators.timer(generate_sub_model)
# update_sub_model = Modules.decorators.timer(update_sub_model)
# gp.Model.optimize = Modules.decorators.timer(gp.Model.optimize)
# %% Generate Initial Feasible Set (Phase 1)
def generate_feasible_sa_list(input_data, init_state_actions):
    state_action_list = init_state_actions
    
    # Initializes
    count = len(state_action_list)-1
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

    # Initialize Sub model
    p1_mast_model, p1_mast_const = generate_phase1_master_model(input_data, mast_model)
    p1_mast_model.Params.LogToConsole = 0
    p1_mast_model.optimize()

    betas = generate_beta_values(input_data, p1_mast_const)
    p1_sub_model, p1_sub_var = generate_sub_model(input_data, betas, True)
    
    # Generates
    while True:
    
        # Generates and Solves Master
        p1_mast_model, p1_mast_const = generate_phase1_master_model(input_data, mast_model)
        p1_mast_model.Params.LogToConsole = 0
        p1_mast_model.Params.method = 2
        p1_mast_model.optimize()
        betas = generate_beta_values(input_data, p1_mast_const)
            
        # Debugging
        print(f'Phase 1 - iteration {count+1}, Mast Objective {p1_mast_model.getObjective().getValue()}')
        count += 1

        # Stops if necessary
        if p1_mast_model.getObjective().getValue() <= 0.00000000001:
            print('Found Feasible Set')
            break

        # Trims
        if (count%100) == 0:
            state_action_list, p1_mast_model, p1_mast_const = trim_sa_list_p1(input_data, state_action_list, p1_mast_model)
        
        # Generates and solves Subproblem 
        p1_sub_model, p1_sub_var = update_sub_model(input_data, p1_sub_model, p1_sub_var, betas, True)
        p1_sub_model.Params.LogToConsole = 0
        p1_sub_model.Params.MIPGap = 1
        p1_sub_model.optimize()
        
        # Update State-Actions
        state_action = generate_state_action(p1_sub_var)
        state_action_list.append(state_action)
        mast_model, mast_var, mast_const = update_master_model(input_data, mast_model, mast_var, mast_const, state_action, count)
        
        # Stops if necessary
        if state_action_list[-1] == state_action_list[-2]:
            print('Unable to find feasible set')
            break

    return state_action_list

# %% Solve the problem (Phase 2)
def generate_optimal_sa_list(input_data, init_state_actions, stabilization_parameter):

    # Initializes
    state_action_list = init_state_actions
    count = len(state_action_list)
    count_same = 1
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

    # Initializes Stabilization Parameters
    beta_avg = generate_beta_estimate(input_data)
    non_neg_count = 0

    # Initialize Sub model
    sub_model, sub_var = generate_sub_model(input_data, beta_avg)

    while True:

        # Generates and Solves Master
        mast_model.Params.LogToConsole = 0
        mast_model.optimize()
        betas = generate_beta_values(input_data, mast_const)

        # Trims
        # if (count%100) == 0:
        #     state_action_list, mast_model, mast_var, mast_const = trim_sa_list(input_data, state_action_list, mast_var)

        if (count_same%100) == 0:
            count_same += 1
            state_action_list, mast_model, mast_var, mast_const = trim_sa_list_p2(input_data, state_action_list, mast_var)

        # Update beta estimate 
        beta_avg = update_beta_estimate(input_data, beta_avg, betas, stabilization_parameter)

        # Generates and solves Subproblem
        sub_model, sub_var = update_sub_model(input_data, sub_model, sub_var, beta_avg)
        sub_model.Params.LogToConsole = 0
        sub_model.optimize()

        # Debugging 
        print(f'Phase 2 - iteration {count+1}, Sub Objective {sub_model.ObjVal:.5E}')
        # Update State-Actions
        if sub_model.ObjVal < 0:
            state_action = generate_state_action(sub_var)
            state_action_list.append(state_action)
            mast_model, mast_var, mast_const = update_master_model(input_data, mast_model, mast_var, mast_const, state_action, count)
        # Same state action
        if state_action_list[-2] == state_action_list[-1]:
            count_same += 1

        # Stopping conditions
        if sub_model.ObjVal >= 0:
            non_neg_count += 1
        if count_same >= 1000:
            print(f'Stuck at {sub_model.ObjVal:.5E}')
            break
        
        if non_neg_count >= 100:
            print('Found Optimal Solution')
            mast_model.optimize()
            betas = generate_beta_values(input_data, mast_const)
            beta_avg = update_beta_estimate(input_data, beta_avg, betas, stabilization_parameter)
            break

        # Adjutst Counts
        count += 1

    return(state_action_list, betas)
# Trim zero state-action pairs occasionally (for p1 model)
def trim_sa_list_p1(input_data, init_state_actions, model):
    initial_len = len(init_state_actions)
    state_action_list = []
    vars = model.getVars()
    
    # Trims
    for var in range(initial_len): 
        if vars[var].x != 0:
            state_action_list.append(init_state_actions[var])

    # Updates The Model
    final_len = len(state_action_list)
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)
    p1_mast_model, p1_mast_const = generate_phase1_master_model(input_data, mast_model)

    print(f'Trimmed SA List - removed {final_len - initial_len}')
        
    return state_action_list, p1_mast_model, p1_mast_const

# Trim zero state-action pairs occasionally (for p2 model)
def trim_sa_list_p2(input_data, init_state_actions, variables):
    
    initial_len = len(init_state_actions)
    state_action_list = []
    
    for sa in range(len(variables)):
        if variables[sa].x != 0:
            state_action_list.append(init_state_actions[sa])

    final_len = len(state_action_list)
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)
    
    print(f'Trimmed SA List - removed {final_len - initial_len}')

    return state_action_list, mast_model, mast_var, mast_const
# Initializes betas for stabilization algorithm
def generate_beta_estimate(input_data):
    indices = input_data.indices

    # Beta Values
    b_0_dual = {}
    b_ul_dual = {}
    b_pw_dual = {}
    b_ps_dual = {}

    # Beta 0
    b_0_dual['b_0'] = 0

    for p in itertools.product(indices['p']):
        # Beta ul
        b_ul_dual[p] = 0

    # Beta pw
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        b_pw_dual[mdc] = 0

    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        b_ps_dual[tmdc] = 0

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ul': b_ul_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas
# Adjusts betas for stabilization algorithm
def update_beta_estimate(input_data, avg_beta, betas, alpha):
    indices = input_data.indices

    # Beta Values
    b_0_dual = {}
    b_ul_dual = {}
    b_pw_dual = {}
    b_ps_dual = {}

    # Beta 0
    b_0_dual['b_0'] = (alpha * avg_beta['b0']['b_0']) + ((1-alpha)*betas['b0']['b_0'])

    for p in itertools.product(indices['p']):
        # Beta ul
        b_ul_dual[p] = (alpha * avg_beta['ul'][p]) + ((1-alpha)*betas['ul'][p])

    # Beta pw
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        b_pw_dual[mdc] = (alpha * avg_beta['pw'][mdc]) + ((1-alpha)*betas['pw'][mdc])

    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        b_ps_dual[tmdc] = (alpha * avg_beta['ps'][tmdc]) + ((1-alpha)*betas['ps'][tmdc])

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ul': b_ul_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas
# %%
