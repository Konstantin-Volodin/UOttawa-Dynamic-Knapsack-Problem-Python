# %%
import optimization 
import simulation as main_sim
from Modules import master_model, data_import, data_export, decorators, simulation

import pandas as pd
import pickle
import os.path
import sys
from matplotlib import pyplot as plt
import contextlib

# Read Data
# my_path = os.path.dirname(__file__)
my_path = os.getcwd()
# print(my_path)
input_data = data_import.read_data(os.path.join(my_path, 'Data', 'simple-data.xlsx'))
# generate_optimal_sa_list = Modules.decorators.timer(generate_optimal_sa_list)


# # %% Sensitivity Analysis
# cw = [1,5,9]
# cc = [1,5,9]
# M = [10, 15, 20]
# gamma = [0.99, 0.95]
# t = [5,6]
# m = [5,6]

# data = {
#     't': [],
#     'm': [],
#     'cw': [],
#     'cc': [],
#     'M': [],
#     'g': [],
#     'b0': [],
#     'bul': [],
#     'pw0':[],
#     'pw1':[],
#     'pw2':[],
#     'pw3':[],
#     'pw4':[],
#     'pw5':[],
#     'pw6':[],
#     'ps10':[],
#     'ps11':[],
#     'ps20':[],
#     'ps21':[],
#     'ps30':[],
#     'ps31':[],
#     'ps40':[],
#     'ps41':[],
#     'ps50':[],
#     'ps51':[],
#     'ps60':[],
#     'ps61':[]
# }

# for ti in t:
#     for mi in m:
#         for cwi in cw:
#             for cci in cc:
#                 for Mi in M:
#                     for gammai in gamma:

#                         input_data.indices['t'] = [i+1 for i in range(ti)]
#                         input_data.indices['m'] = [i for i in range(mi+1)]
#                         input_data.model_param.cw['P1'] = cwi
#                         input_data.model_param.cc['P1'] = cci
#                         input_data.model_param.M = Mi
#                         input_data.model_param.gamma = gammai

#                         for i in input_data.indices['t']:
#                             input_data.model_param.cs['P1'][i] = input_data.model_param.cs['P1'][i-1] + input_data.model_param.cw['P1'] * (input_data.model_param.gamma**i)

#                         # Phase 1
#                         with contextlib.redirect_stdout(None):
#                             init_state, init_action = master_model.initial_sa(input_data)
#                             state_action_list = [(init_state, init_action)]
#                             feasible_list, betas = optimization.p1_algo(input_data, state_action_list)

#                         # Phase 2
#                         with contextlib.redirect_stdout(None):
#                             stabilization_parameter = 0.3
#                             optimal_list, betas = optimization.p2_algo(input_data, feasible_list,stabilization_parameter)

#                         data['t'].append(ti)
#                         data['m'].append(mi)
#                         data['cw'].append(cwi)
#                         data['cc'].append(cci)
#                         data['M'].append(Mi)
#                         data['g'].append(gammai)
#                         data['b0'].append(betas['b0']['b_0'])
#                         data['bul'].append(betas['ul'][('PPE1',)])
#                         data['pw0'].append(betas['pw'][(0, 'C1', 'P1','CPU1')])
#                         data['pw1'].append(betas['pw'][(1, 'C1', 'P1','CPU1')])
#                         data['pw2'].append(betas['pw'][(2, 'C1', 'P1','CPU1')])
#                         data['pw3'].append(betas['pw'][(3, 'C1', 'P1','CPU1')])
#                         data['pw4'].append(betas['pw'][(4, 'C1', 'P1','CPU1')])
#                         data['pw5'].append(betas['pw'][(5, 'C1', 'P1','CPU1')])
#                         if mi == 6:
#                             data['pw6'].append(betas['pw'][(6, 'C1', 'P1','CPU1')])
#                         else:
#                             data['pw6'].append(None)
#                         data['ps10'].append(betas['ps'][(1, 0, 'C1', 'P1','CPU1')])
#                         data['ps11'].append(betas['ps'][(1, 1, 'C1', 'P1','CPU1')])
#                         data['ps20'].append(betas['ps'][(2, 0, 'C1', 'P1','CPU1')])
#                         data['ps21'].append(betas['ps'][(2, 1, 'C1', 'P1','CPU1')])
#                         data['ps30'].append(betas['ps'][(3, 0, 'C1', 'P1','CPU1')])
#                         data['ps31'].append(betas['ps'][(3, 1, 'C1', 'P1','CPU1')]) 
#                         data['ps40'].append(betas['ps'][(4, 0, 'C1', 'P1','CPU1')])
#                         data['ps41'].append(betas['ps'][(4, 1, 'C1', 'P1','CPU1')])
#                         data['ps50'].append(betas['ps'][(5, 0, 'C1', 'P1','CPU1')])
#                         data['ps51'].append(betas['ps'][(5, 1, 'C1', 'P1','CPU1')])
#                         if ti == 6:
#                             data['ps60'].append(betas['ps'][(6, 0, 'C1', 'P1','CPU1')])
#                             data['ps61'].append(betas['ps'][(6, 1, 'C1', 'P1','CPU1')])
#                         else:
#                             data['ps60'].append(None)
#                             data['ps61'].append(None)

#                         print('Done')

# df = pd.DataFrame(data)
# with open(os.path.join(my_path, 'Data', 'sensitivity analysis', f'model_param.pkl'), 'wb') as outp:
#     pickle.dump(df, outp, pickle.HIGHEST_PROTOCOL)

# df = pd.read_pickle(os.path.join(my_path, 'Data', 'sensitivity analysis', f'model_param.pkl'))

# %% Optimization
# Phase 1
init_state, init_action = master_model.initial_sa(input_data)
state_action_list = [(init_state, init_action)]
feasible_list, betas = optimization.p1_algo(input_data, state_action_list)
data_export.export_betas(betas, os.path.join(my_path, 'Data', f'simple-feasible.xlsx'))
with open(os.path.join(my_path, 'Data', f'simple-feasible.pkl'), 'wb') as outp:
    pickle.dump(feasible_list, outp, pickle.HIGHEST_PROTOCOL)

# Phase 2
stabilization_parameter = 0.9
optimal_list, betas = optimization.p2_algo(input_data, feasible_list,stabilization_parameter)
data_export.export_betas(betas, os.path.join(my_path, 'Data', f'simple-optimal.xlsx'))
with open(os.path.join(my_path, 'Data', f'simple-optimal.pkl'), 'wb') as outp:
    pickle.dump(optimal_list, outp, pickle.HIGHEST_PROTOCOL)
# %% Compare Policies
# Import betas
fig, axes = plt.subplots(1, 1)
betas = data_import.read_betas(os.path.join(my_path, 'Data', f'simple-optimal.xlsx'))
main_sim.compare_policies(input_data, betas, 2, 2000, 1000, axes)
fig.savefig(os.path.join(my_path, 'Data', f'simple-optimal.pdf'))

# %% Test out policies
# input_data.arrival[('C1', 'P1', 'CPU1')] = 20
betas = data_import.read_betas(os.path.join(my_path, 'Data', f'simple-optimal.xlsx'))
# main_sim.test_out_policy(input_data, 100, simulation.mdp_policy, "MDP", betas)
# main_sim.test_out_policy(input_data, 10, simulation.myopic_policy, "Myopic")
# %%
