# %% Packages
from Modules.data_import import *
from Modules.data_export import *
from Modules.master_model import *
from Modules.sub_problem import *
from Modules.decorators import timer
import os.path

# read_data = timer(read_data)
# generate_master_model = timer(generate_master_model)
# generate_phase1_master_model = timer(generate_phase1_master_model)
# generate_sub_model = timer(generate_sub_model)
# update_master_model = timer(update_master_model)
# gp.Model.optimize = timer(gp.Model.optimize)

my_path = os.path.dirname(__file__)
input_data = read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))

# %% Generate Initial Feasible Set (Phase 1)
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]

count = 0
mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

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
    if p1_mast_model.getObjective().getValue() <= 0:
        print('Found Feasible Set')
        break
    
    # Generates and solves Subproblem 
    p1_sub_model, p1_sub_var = generate_sub_model(input_data, betas, True)
    p1_sub_model.Params.LogToConsole = 0
    # p1_sub_model.setParam('Presolve', 0)
    # p1_sub_model.setParam('MIPGap', 0)
    p1_sub_model.optimize()
    
    # Update State-Actions
    state_action = generate_state_action(p1_sub_var)
    state_action_list.append(state_action)
    mast_model, mast_var, mast_const = update_master_model(input_data, mast_model, mast_var, mast_const, state_action, count)
    
    # Stops if necessary
    if state_action_list[-1] == state_action_list[-2]:
        print('Unable to find feasible set')
        break

# %% Solve the problem (Phase 2)
count = 0
mast_model, mast_var, mast_const = generate_master_model(input_data, state_action_list)

while True:

    # Generates and Solves Master
    mast_model.Params.LogToConsole = 0
    mast_model.optimize()
    betas = generate_beta_values(input_data, mast_const)

    # Generates and solves Subproblem
    sub_model, sub_var = generate_sub_model(input_data, betas)
    sub_model.Params.LogToConsole = 0
    sub_model.optimize()

    # Debugging 
    print(f'Phase 2 - iteration {count+1}, Sub Objective {sub_model.getObjective().getValue()}')
    count += 1

    # Stops if necessary
    if sub_model.ObjVal >= 0:
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
# %% Misc Code
# Save state-actions
my_path = os.path.dirname(__file__)
export_all_state_action(state_action_list, os.path.join(my_path, 'Data', 'SA-Pairs.xlsx'))


# p1_mast_model.write('mast_p1.lp')
# p1_sub_model.write('sub_prob_p1.lp')

# mast_model.write('mast_p2.lp')
# sub_model.write('sub_prob_p2.lp')
# %%
