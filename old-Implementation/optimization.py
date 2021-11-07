# %% Packages
from Modules.data_import import *
from Modules.data_export import *
from Modules.master_model import *
from Modules.sub_problem import *    

# %% Generate Initial Feasible Set (Phase 1)
def generate_feasible_sa_list(input_data, init_state_actions):
    state_action_list = init_state_actions
    
    # Initializes
    count = len(state_action_list)-1
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

    # Generates
    while True:
    
        # Generates and Solves Master
        p1_mast_model, p1_mast_const = generate_phase1_master_model(input_data, mast_model)
        p1_mast_model.Params.LogToConsole = 0
        p1_mast_model.optimize()
        betas = generate_beta_values(input_data, p1_mast_const)
            
        # Debugging
        print(f'Phase 1 - iteration {count+1}, Mast Objective {p1_mast_model.getObjective().getValue()}')
        count += 1

        # Stops if necessary
        if p1_mast_model.getObjective().getValue() <= 0.00000000001:
            print('Found Feasible Set')
            break
        
        # Generates and solves Subproblem 
        p1_sub_model, p1_sub_var = generate_sub_model(input_data, betas, True)
        p1_sub_model.Params.LogToConsole = 0
        p1_sub_model.optimize()
        
        # Update State-Actions
        state_action = generate_state_action(p1_sub_var)
        state_action_list.append(state_action)
        mast_model, mast_var, mast_const = update_master_model(input_data, mast_model, mast_var, mast_const, state_action, count)
        
        # Stops if necessary
        if state_action_list[-1] == state_action_list[-2]:
            print('Unable to find feasible set')
            break

    return(state_action_list)

# %% Solve the problem (Phase 2)
def generate_optimal_sa_list(input_data, init_state_actions):

    # Initializes
    state_action_list = init_state_actions
    count = len(state_action_list)
    count_without_removal = 0
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

    while True:

        # Generates and Solves Master
        mast_model.Params.LogToConsole = 0
        mast_model.optimize()
        betas = generate_beta_values(input_data, mast_const)

        # Drops state action pairs as needed
        if count_without_removal >= 50:
            state_action_list, mast_model, mast_var, mast_const = trim_sa_list(input_data, state_action_list, mast_var)
            count_without_removal = 0

        # Generates and solves Subproblem
        sub_model, sub_var = generate_sub_model(input_data, betas)
        sub_model.Params.LogToConsole = 0
        sub_model.optimize()

        # Debugging 
        print(f'Phase 2 - iteration {count+1}, Sub Objective {sub_model.getObjective().getValue()}')


        # Stops if necessary
        if sub_model.ObjVal >= -0.00000000000000001:
            print('Found Optimal Solution')
            break

        # Update State-Actions
        state_action = generate_state_action(sub_var)
        state_action_list.append(state_action)
        mast_model, mast_var, mast_const = update_master_model(input_data, mast_model, mast_var, mast_const, state_action, count)

        # Stops if necessary
        if state_action_list[-1] == state_action_list[-2]:
            print('Unable to find feasible set')
            break

        count += 1
        count_without_removal += 1

    return(state_action_list, betas)
# Trim zero state-action pairs occasionally
def trim_sa_list(input_data, init_state_actions, variables):
    
    initial_len = len(init_state_actions)
    state_action_list = []
    
    for sa in range(len(variables)):
        if variables[sa].x != 0:
            state_action_list.append(init_state_actions[sa])

    final_len = len(state_action_list)
    mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)
    print(f'Trimmed SA List - removed {final_len - initial_len}')

    return(state_action_list, mast_model, mast_var, mast_const)


# %% Misc Code
# Write the model
# p1_mast_model.write('mast_p1.lp')
# p1_sub_model.write('sub_prob_p1.lp')

# mast_model.write('mast_p2.lp')
# sub_model.write('sub_prob_p2.lp')