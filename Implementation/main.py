# %% Packages
from Modules.data_import import *
from Modules.data_export import *
from Modules.master_model import *
from Modules.sub_problem import *
from Modules.decorators import timer
import os.path

# %% Reads Data
my_path = os.path.dirname(__file__)
read_data = timer(read_data)
input_data = read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))

# %% Generate Initial Feasible Set
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]

count = 0
for i in range(100):
    # Adjusted Master Model
    p1_mast_model, p1_mast_var, p1_mast_const = generate_phase1_master_model(input_data, state_action_list)
    p1_mast_model.Params.LogToConsole = 0
    p1_mast_model.optimize()
    print(f'iteration {count}, {p1_mast_model.getObjective().getValue()}')
    count += 1
    betas = generate_beta_values(input_data, p1_mast_const)
    if p1_mast_model.getObjective().getValue() <= 0:
        break
    # Adjusted Subproblem
    p1_sub_model, p1_sub_var = generate_sub_model(input_data, betas, True)
    p1_sub_model.Params.LogToConsole = 0
    p1_sub_model.optimize()
    state_action = generate_state_action(p1_sub_var)
    # Adjusts state action set
    state_action_list.append(state_action)

# %% Saves Feasible State-Action pairs
my_path = os.path.dirname(__file__)
export_all_state_action(state_action_list, os.path.join(my_path, 'Data', 'SA-Pairs.xlsx'))


# %% Generate Sub Model (Phase 2)
sub_model, sub_variables = generate_sub_model(input_data, betas)
sub_model.write('sub_prob_p2.lp')

# %% Generate Master Model (Phase 2)
mast_model, variables, constraints = generate_master_model(input_data, state_action_list)
mast_model.write('mast_p2.lp')


# %% Solve the problem (Phase 2?)
generate_initial_state_action = timer(generate_initial_state_action)
generate_master_model = timer(generate_master_model)

init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]
mast_model, variables, constraints = generate_master_model(input_data, state_action_list)
mast_model.write('mast.lp')
for i in range(10):

    # Solves Updated Master Problem
    mast_model = generate_master_model(input_data, state_action_list)
    mast_model.optimize()
    mast_model.write(f'mast{i}.lp')

    # Finds Violated Constraint
    # subprolem_solution = solve_pricing_problem(model)
    # if subprolem_solution.val <= 0: 
    #     break
    # else:
    #     state_action_list.append((subproblem_solution.state, subproblem_solution.action))

    state_action_list.append((init_state, init_action))
    # update_master_model(model, subprolem_solution.state_action)

# final_beta = model.beta


# %%
