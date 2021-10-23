# %% Packages
# from Modules import data_import, data_export
from Modules.data_import import *
from Modules.data_export import *
from Modules import master_model
from Modules import sub_problem
import itertools
import gurobipy as gp
import os.path

# %% Generate Initial Feasible Set (Phase 1)
def p1_algo(input_data, init_state_actions):
    
    # Initializes
    state_action_list = init_state_actions
    count = len(state_action_list)-1
    mast_model, mast_var, mast_const = master_model.master_p2(input_data, state_action_list)

    # Initialize Sub model
    p1_mast_model, p1_mast_const = master_model.master_p1(input_data, mast_model)
    p1_mast_model.Params.LogToConsole = 0
    p1_mast_model.optimize()

    betas = master_model.get_betas(input_data, p1_mast_const)
    p1_sub_model, p1_sub_var = sub_problem.subproblem(input_data, betas, True)
    
    # Algorithm
    while True:
    
        # Generates and Solves Master
        p1_mast_model, p1_mast_const = master_model.master_p1(input_data, mast_model)
        p1_mast_model.Params.LogToConsole = 0
        p1_mast_model.optimize()
        p1_mast_model.write('m_p1.lp')
        betas = master_model.get_betas(input_data, p1_mast_const)
            
        # Debugging
        print(f'Phase 1 - iteration {count+1}, Mast Objective {p1_mast_model.getObjective().getValue()}')
        count += 1

        # Stops if successful
        if p1_mast_model.getObjective().getValue() <= 0.00000000001:
            p1_mast_model.optimize()
            betas = master_model.get_betas(input_data, p1_mast_const)
            print('Found Feasible Set')
            break

        # Trims
        if (count%100) == 0:
            state_action_list, p1_mast_model, p1_mast_const = trim_sa_list_p1(input_data, state_action_list, p1_mast_model)
        
        # Generates and solves Subproblem 
        p1_sub_model, p1_sub_var = sub_problem.update_sub(input_data, p1_sub_model, p1_sub_var, betas, True)
        p1_sub_model.Params.LogToConsole = 0
        p1_sub_model.optimize()
        p1_sub_model.write('p1_s.lp')
        
        # Update State-Actions
        state_action = sub_problem.get_sa(p1_sub_var)
        state_action_list.append(state_action)
        mast_model, mast_var, mast_const = master_model.update_master(input_data, mast_model, mast_var, mast_const, state_action, count)
        
        # Stops if unsuccessful
        if state_action_list[-1] == state_action_list[-2]:
            print('Unable to find feasible set')
            break

    return state_action_list, betas

# %% Solve the problem (Phase 2)
def p2_algo(input_data, init_state_actions, stabilization_parameter, start=False):

    # Initializes
    state_action_list = init_state_actions
    count = len(state_action_list) - 1
    count_same = 1
    mast_model, mast_var, mast_const = master_model.master_p2(input_data, state_action_list)

    # Initializes Stabilization Parameters
    beta_avg = 0
    if start == False: beta_avg = generate_beta_estimate(input_data)
    else: beta_avg = start

    non_neg_count = 0

    # Initialize Sub model
    sub_model, sub_var = sub_problem.subproblem(input_data, beta_avg)

    while True:

        # Generates and Solves Master
        mast_model.Params.LogToConsole = 0
        mast_model.optimize()
        betas = master_model.get_betas(input_data, mast_const)

        # Update beta estimate 
        beta_avg = update_beta_estimate(input_data, beta_avg, betas, stabilization_parameter)

        # Trims
        if (count%300) == 0:
            state_action_list, mast_model, mast_var, mast_const = trim_sa_list_p2(input_data, state_action_list, mast_var)

        if (count_same%100) == 0:
            count_same += 1
            state_action_list, mast_model, mast_var, mast_const = trim_sa_list_p2(input_data, state_action_list, mast_var)

        # Generates and solves Subproblem
        sub_model, sub_var = sub_problem.update_sub(input_data, sub_model, sub_var, beta_avg)
        sub_model.Params.LogToConsole = 0
        sub_model.optimize()

        # Debugging 
        print(f'Phase 2 - iteration {count+1}, Sub Objective {sub_model.ObjVal:.5E}')

        # Update State-Actions
        if sub_model.ObjVal < 0:
            state_action = sub_problem.get_sa(sub_var)
            state_action_list.append(state_action)
            mast_model, mast_var, mast_const = master_model.update_master(input_data, mast_model, mast_var, mast_const, state_action, count)

        # Same state action
        if state_action_list[-2] == state_action_list[-1]:
            count_same += 1
            
        # Stops if unsuccessful
        if sub_model.ObjVal >= 0:
            non_neg_count += 1
        if count_same >= 1000:
            print(f'Stuck at {sub_model.ObjVal:.5E}')
            break
        
        # Stops if successful
        if non_neg_count >= 100:
            print('Found Optimal Solution')
            mast_model.optimize()
            betas = master_model.get_betas(input_data, mast_const)
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
    mast_model, mast_var, mast_const = master_model.master_p2(input_data, state_action_list)
    p1_mast_model, p1_mast_const = master_model.master_p1(input_data, mast_model)

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
    mast_model, mast_var, mast_const = master_model.master_p2(input_data, state_action_list)
    
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

    # Beta ul
    for p in itertools.product(indices['p']):
        b_ul_dual[p] = 0

    # Beta pw
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        b_pw_dual[mdkc] = 0

    # Beta ps
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        b_ps_dual[tmdkc] = 0

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

    # Beta ul
    for p in itertools.product(indices['p']):
        b_ul_dual[p] = (alpha * avg_beta['ul'][p]) + ((1-alpha)*betas['ul'][p])

    # Beta pw
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        b_pw_dual[mdkc] = (alpha * avg_beta['pw'][mdkc]) + ((1-alpha)*betas['pw'][mdkc])

    # Beta ps
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        b_ps_dual[tmdkc] = (alpha * avg_beta['ps'][tmdkc]) + ((1-alpha)*betas['ps'][tmdkc])

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ul': b_ul_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas
# %%
