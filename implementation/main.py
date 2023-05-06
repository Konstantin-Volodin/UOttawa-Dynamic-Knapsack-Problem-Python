#%% INIT
# INITIALIZATION AND DATA PARAMETERS
from modules.optimization_new import optimization_handler
from modules.simulation_new import simulation_handler

# MODEL DATA
test_modifier = ''
data_type = 'full-sm'

#%% OPTIM
# OPTIMIZATION PARAMETERS
optim_param = {'iterations': 500,
               'beta_function': [(0, 0.8),
                                 (3000, 0.99)],
               'subproblem_mip_gap': 0.001}
optim_paths = {'import_params': f'data/{data_type}/input/{test_modifier}{data_type}-np-dt.xlsx',
               'export_betas': f'data/{data_type}/input/betas/{test_modifier}{data_type}-np-opt.pkl',
               'export_model': f'data/{data_type}/input/model/{test_modifier}{data_type}-np-md.lp'}

# EXECUTE OPTIMIZER
optimizer = optimization_handler(optim_params = optim_param, 
                                 optim_paths = optim_paths)
optimizer.read_data()
optimizer.cuu = 0       # NO R3
# optimizer.generate_master()
# optimizer.generate_subproblem()
# optimizer.solve_phase1()
# optimizer.solve_phase2()
# optimizer.save_data()

#%% SIM
# SIMULATION PARAMETERS
sim_param = {'replications': 30,
             'warm_up': 10,
             'duration': 100}
sim_paths = {'import_params': f'data/{data_type}/input/{test_modifier}{data_type}-np-dt.xlsx',
             'import_betas': f'data/{data_type}/input/betas/{test_modifier}{data_type}-np-opt.pkl',
             'export_summary_costs': f'data/{data_type}/res/{test_modifier}{data_type}-res.txt',
             'export_summary_picture': f'data/{data_type}/res/{test_modifier}{data_type}-res.html',
             'export_state_myopic': f'data/{data_type}/res/state/{test_modifier}{data_type}-res-my.txt',
             'export_state_mdp': f'data/{data_type}/res/state/{test_modifier}{data_type}-res-mdp.txt',
             'export_cost_myopic': f'data/{data_type}/res/cost/{test_modifier}{data_type}-res-my.txt',
             'export_cost_mdp': f'data/{data_type}/res/cost/{test_modifier}{data_type}-res-mdp.txt',
             'export_util_myopic': f'data/{data_type}/res/util/{test_modifier}{data_type}-res-my.txt',
             'export_util_mdp': f'data/{data_type}/res/util/{test_modifier}{data_type}-res-mdp.txt',
             'export_sa_myopic': f'data/{data_type}/res/sa/{test_modifier}{data_type}-res-my.txt',
             'export_sa_mdp': f'data/{data_type}/res/sa/{test_modifier}{data_type}-res-mdp.txt',}

# EXECUTE SIMULATOR
simulator = simulation_handler(sim_param = sim_param,
                               sim_paths = sim_paths)
simulator.read_data()
simulator.cuu = 0       # NO R3
simulator.generate_myopic()
simulator.generate_mdp()

simulator.simulation_myopic()
# simulator.simulation_mdp()
# %%
