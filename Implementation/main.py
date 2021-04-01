# %% Packages
from Modules.data_import import *
from Modules.master_model import *
from Modules.sub_problem import *
from Modules.decorators import timer
import os.path

# %% Reads Data
my_path = os.path.dirname(__file__)
read_data = timer(read_data)
input_data = read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))

# %% Generate Master Model (Phase 1)
init_state, init_action = generate_initial_state_action(input_data)
state_action_list = [(init_state, init_action)]
p1_mast_model, p1_variables, p1_constraints = generate_phase1_master_model(input_data, state_action_list)
p1_mast_model.write('mast_p1.lp')

p1_mast_model.optimize()
betas = generate_beta_values(input_data, p1_constraints)
# %% Generate Sub Model (Phase 2)
sub_model = generate_sub_model(input_data, betas)
sub_model.write('sub_prob.lp')


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
