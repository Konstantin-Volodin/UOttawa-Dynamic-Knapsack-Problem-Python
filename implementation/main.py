#%% INIT
# INITIALIZATION AND DATA PARAMETERS
from modules.optimization_new import optimization_handler
from modules.simulation_new import simulation_handler

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
        optim_paths = {'import_params': f'data/{data_type}/input/{test_modifier}{data_type}-np-dt.xlsx',
                    'export_betas': f'data/{data_type}/input/betas/{test_modifier}{data_type}-np-opt-{cuu_sn}-{or_sn}.pkl',
                    'export_model': f'data/{data_type}/input/model/{test_modifier}{data_type}-np-md-{cuu_sn}-{or_sn}.lp'}


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
        sim_paths = {'import_params': f'data/{data_type}/input/{test_modifier}{data_type}-np-dt.xlsx',
                    'import_betas': f'data/{data_type}/input/betas/{test_modifier}{data_type}-np-opt-{cuu_sn}-{or_sn}.pkl',
                    'export_summary_costs': f'data/{data_type}/res/{test_modifier}{data_type}-res-{cuu_sn}-{or_sn}.txt',
                    'export_summary_picture': f'data/{data_type}/res/{test_modifier}{data_type}-res-{cuu_sn}-{or_sn}.html',
                    'export_state_myopic': f'data/{data_type}/res/state/{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt',
                    'export_state_mdp': f'data/{data_type}/res/state/{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt',
                    'export_cost_myopic': f'data/{data_type}/res/cost/{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt',
                    'export_cost_mdp': f'data/{data_type}/res/cost/{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt',
                    'export_util_myopic': f'data/{data_type}/res/util/{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt',
                    'export_util_mdp': f'data/{data_type}/res/util/{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt',
                    'export_sa_myopic': f'data/{data_type}/res/sa/{test_modifier}{data_type}-res-my-{cuu_sn}-{or_sn}.txt',
                    'export_sa_mdp': f'data/{data_type}/res/sa/{test_modifier}{data_type}-res-mdp-{cuu_sn}-{or_sn}.txt',}

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
