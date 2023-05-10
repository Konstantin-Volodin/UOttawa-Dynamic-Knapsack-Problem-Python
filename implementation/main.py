#%% INIT
# INITIALIZATION AND DATA PARAMETERS
from modules.optimization_new import optimization_handler
from modules.simulation_new import simulation_handler
import os.path


# MODEL DATA
test_modifier = ''
data_type = 'full-sm'


# #%% PARAMS
cuu_sens = [0, 1000]
or_sens = [1, 1.1, 1.2, 1.3]
 
for cuu_sn in cuu_sens:
    for or_sn in or_sens:


        # OPTIMIZATION PARAMETERS
        optim_param = {'iterations': 50000,
                    'beta_function': [(0, 0.9),
                                      (3000, 0.99)],
                    'subproblem_mip_gap': 0.1}
        optim_paths = {'import_params': os.path.join('data',data_type,'input',f'{test_modifier}{data_type}-np-dt.xlsx'),
                       'export_betas': os.path.join('data',data_type, 'input', 'betas', f'{test_modifier}{data_type}-np-opt-{cuu_sn}-{or_sn}.pkl'),
                       'export_model': os.path.join('data',data_type, 'input', 'model', f'{test_modifier}{data_type}-np-opt-{cuu_sn}-{or_sn}.lp'),}


        # EXECUTE OPTIMIZER
        optimizer = optimization_handler(optim_params = optim_param, 
                                        optim_paths = optim_paths)
        optimizer.read_data()
        optimizer.cuu = cuu_sn
        optimizer.input_data.ppe_data['OR_Time'].expected_units *= or_sn
        optimizer.generate_master()
        optimizer.generate_subproblem()
        optimizer.solve_phase1()
        optimizer.solve_phase2()
        optimizer.save_data()


        # SIMULATION PARAMETERS
        sim_param = {'replications': 30,
                    'warm_up': 250,
                    'duration': 1000}
        sim_paths = {'import_params': os.path.join('data',data_type,'input',f'{test_modifier}{data_type}-np-dt.xlsx'),
                     'import_betas': os.path.join('data',data_type, 'input', 'betas', f'{test_modifier}{data_type}-np-opt-{cuu_sn}-{or_sn}.pkl'),
                     'export_summary_costs': os.path.join('data',data_type,'res',f'{test_modifier}{data_type}-res-{cuu_sn}-{or_sn}.txt'),
                     'export_summary_picture': os.path.join('data',data_type,'res',f'{test_modifier}{data_type}-res-{cuu_sn}-{or_sn}.html'),
                     'export_state_myopic': os.path.join('data',data_type,'res','state',f'{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt'),
                     'export_state_mdp': os.path.join('data',data_type,'res','state',f'{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt'),
                     'export_cost_myopic': os.path.join('data',data_type,'res','cost',f'{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt'),
                     'export_cost_mdp': os.path.join('data',data_type,'res','cost',f'{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt'),
                     'export_util_myopic': os.path.join('data',data_type,'res','util',f'{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt'),
                     'export_util_mdp': os.path.join('data',data_type,'res','util',f'{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt'),
                     'export_sa_myopic': os.path.join('data',data_type,'res','sa',f'{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt'),
                     'export_sa_mdp': os.path.join('data',data_type,'res','sa',f'{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt'),}

        # EXECUTE SIMULATOR
        simulator = simulation_handler(sim_param = sim_param,
                                       sim_paths = sim_paths)
        simulator.read_data()
        simulator.cuu = cuu_sn
        simulator.input_data.ppe_data['OR_Time'].expected_units *= or_sn
        simulator.generate_myopic()
        simulator.generate_mdp()
        simulator.simulation_myopic()
        simulator.simulation_mdp()
        simulator.save_data()
# %%
