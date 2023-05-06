#%%
from modules.optimization_new import optimization_handler
# import modules.simulation_new
import time

# MODEL DATA
test_modifier = ''
data_type = 'full-sm'

# OPTIMIZATION PARAMETERS
optim_param = {'iterations': 100000,
               'beta_function': [(0, 0.8),
                                 (3000, 0.99)],
               'subproblem_mip_gap': 0.001}
optim_paths = {'import_params': f'data/{data_type}/data/{test_modifier}{data_type}-np-dt.xlsx',
               'export_betas': f'data/{data_type}/data/betas/{test_modifier}{data_type}--np-opt.pkl',
               'export_model': f'data/{data_type}/data/model/{test_modifier}{data_type}-np-md.lp'}

# EXECUTE OPTIMIZER
optimizer = optimization_handler(optim_params = optim_param, 
                                 optim_paths = optim_paths)
optimizer.read_data()
optimizer.cuu = 0       # NO R3
optimizer.generate_master()
optimizer.generate_subproblem()
optimizer.solve_phase1()
optimizer.solve_phase2()




#%%
# SIMULATION PARAMETERS
sim_param = {'replications': 30,
             'warm_up': 250,
             'duration': 1000}
sim_paths = {'import_params': f'data/sens-data/{data_type}/{test_modifier}{data_type}-np-dt.xlsx',
             'import_betas': f'data/sens-data/{data_type}/betas/{test_modifier}{data_type}--np-opt.pkl',
             'export_summary_costs': f'data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res-nopri-.txt',
             'export_summary_picture': f'data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res-nopri-R1R2.html',
             'export_state_myopic': f'',
             'export_state_mdp': f'',
             'export_cost_myopic': f'',
             'export_cost_mdp': f'',
             'export_util_myopic': f'',
             'export_util_mdp': f'',
             'export_sa_myopic': f'',
             'export_sa_mdp': f'',}



import_data_sim = import_data_opt
import_betas_sim = export_data_opt
export_txt_sim = f'Data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res-nopri-R1R2.txt'
export_pic_sim = f'Data/sens-res/{data_type}/{test_modifier}{data_type}-optimal-res-nopri-R1R2.html'
export_state_my = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-state-my-nopri-R1R2.txt'
export_state_md = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-state-md-nopri-R1R2.txt'
export_cost_my = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-cost-my-nopri-R1R2.txt'
export_cost_md = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-cost-md-nopri-R1R2.txt'
export_util_my = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-util-my-nopri-R1R2.txt'
export_util_md = f'Data/sens-res/{data_type}/state-action/{test_modifier}{data_type}-util-md-nopri-R1R2.txt'
export_sa_my = f'Data/sens-res/{data_type}/logging/{test_modifier}{data_type}-sa-my-nopri-R1R2.txt'
export_sa_md = f'Data/sens-res/{data_type}/logging/{test_modifier}{data_type}-sa-md-nopri-R1R2.txt'

# import_data_p2 = export_data_p2




# Simulation Parameters
replications = 3
warm_up = 1000
duration = 3000
show_policy = False
import_data_sim = import_data_opt
import_betas_sim = export_data_opt
export_txt_sim = f'sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.txt'
export_pic_sim = f'sens-res/{data_type}/{test_modifier}{data_type}-optimal-res.html'

# # Execute
# start_time_opt = time.time()
# optimization_new.main_func(iter_lims, beta_fun, sub_mip_gap, import_data_opt, export_data_opt, export_data_p2)
# end_time_opt = time.time()
# simulation_new.main_func(replications, warm_up, duration, show_policy, import_data_sim, import_betas_sim, export_txt_sim, export_pic_sim)
# end_time_sim = time.time()
# print(f'{test_modifier}{data_type}\tOptimization: {end_time_opt-start_time_opt} sec \tSimulation: {end_time_sim-end_time_opt} sec')
# print(f'{beta_fun}')