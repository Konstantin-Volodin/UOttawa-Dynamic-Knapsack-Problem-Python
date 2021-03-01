# %% Packages
from Modules import data_import

import os.path
import itertools
import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import gurobipy as gp
from gurobipy import GRB


# %% Reads Data
my_path = os.path.dirname(__file__)
indices, ppe_data, usage, arrival, transition, model_param = data_import.read_data(os.path.join(my_path, 'Data', 'Data.xlsx'))
M = 100
gamma = 0.9


#%% Defining states and actions
@dataclass(frozen=True)
class state:
    ue_tp: Dict[ Tuple[str],float ] # units expected on time t, ppe p
    uu_tp: Dict[ Tuple[str],float ] # units used on time t, ppe p
    uv_tp: Dict[ Tuple[str],float ] # units violated on time t, ppe p
    pe_dc: Dict[ Tuple[str],float ] # new patient arrivals of complexity d, cpu c
    pw_mdc: Dict[ Tuple[str],float ] # patients waiting for m periods, of complexity d, cpu c
    ps_tmdc: Dict[ Tuple[str],float ] # patients scheduled into time t, who have waited for m periods, of complexity d, cpu c

@dataclass(frozen=True)
class action:
    sc_tmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods to schedule into t (m of 0 corresponds to pe)
    rsc_ttpmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods, to reschedule from t to tp 

# %% initial state-action for testing
# Patient States
pe = {}
for dc in itertools.product(indices['d'], indices['c']):
    pe[dc] = arrival[dc]
pw = {}
for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
    if mdc[0] == 0: continue
    pw[mdc] = 0
ps = {}
for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
    ps[tmdc] = 0

# Unit States
ue = {}
for tp in itertools.product(indices['t'], indices['p']):
    expected_units = ppe_data[tp[1]]['expected units']
    ue[tp] = expected_units
uu = {}
for tp in itertools.product(indices['t'], indices['p']):
    uu[tp] = 0
uv = {}
for tp in itertools.product(indices['t'], indices['p']):
    uv[tp] = np.max((0, uu[tp] - ue[tp]))

# Actions
sc = {}
for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
    sc[tmdc] = 0
rsc = {}
for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
    if ttpmdc[0] != 1: rsc[ttpmdc] = 0
    elif ttpmdc[1] != 1: rsc[ttpmdc] = 0
    else: rsc[ttpmdc] = 0

test_state = state(ue, uu, uv, pe, pw, ps)
test_action = action(sc, rsc)

#%% initial expected parameters for testing
E_uu = test_state.uu_tp
E_ue = test_state.ue_tp
E_pe = test_state.pe_dc
E_pw = test_state.pw_mdc
E_ps = test_state.ps_tmdc

# %% Various Master Model Functions
@dataclass(frozen=True)
class constraint_parameter:
    lhs_param: Dict[ Tuple[str],float ]
    rhs_param: Dict[ Tuple[str],float ]
    sign: Dict[ Tuple[str],str ]
    name:  Dict[ Tuple[str],str ]

# Cost Function
def cost_function(state: state, action: action) -> constraint_parameter:
    cost = 0

    # Cost of Waiting
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):

        # sum_t sc part
        psc = 0 
        for t in indices['t']:
            psc += action.sc_tmdc[(t, mdc[0], mdc[1], mdc[2])]
        
        # if m = 0, uses unscheduled patients who just arrived
        # if m > 0, uses patients waiting for m
        if mdc[0] == 0: cost += model_param['cw']**mdc[0] * ( state.pe_dc[(mdc[1],mdc[2])] - psc )
        else: cost += model_param['cw']**mdc[0] * ( state.pw_mdc[mdc] - psc )

    # Cost of Cancelling
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] > ttpmdc[1]: #good schedule
            cost -= model_param['cc'] * test_action.rsc_ttpmdc[ttpmdc]
        elif ttpmdc[1] > ttpmdc[0]: #bad schedule
            cost += model_param['cc'] * test_action.rsc_ttpmdc[ttpmdc]

    # Violating unit bounds
    for tp in itertools.product(indices['t'], indices['p']):
        cost += state.uv_tp[tp] * M

    return(cost)

# Generates beta 0 constraint data 
def b_0_constraint(state: state, action: action) -> constraint_parameter:

    constr_data = constraint_parameter(
        {"b_0": 1-gamma}, 
        {"b_0": 1},
        {"b_0": '='},
        {"b_0": 'beta_0'},
    )
    return constr_data

# Generates beta ue constraint data
def b_ue_constraint(state: state, action: action) -> constraint_parameter:
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):
        lhs[tp] = state.ue_tp[tp] - gamma*(ppe_data[tp[1]]['expected units'])
        rhs[tp] = E_ue[tp]
        sign[tp] = ">="
        name[tp] = f"beta_ue_tp_{tp[0]}_{tp[1].replace(' ', '-')}"

    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data

# Generates beta uu constraint data
def b_uu_constraint(state: state, action: action) -> constraint_parameter:
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == indices['t'][-1]:
            lhs[tp] = state.uu_tp[tp]
        else:
            parameter_val = state.uu_tp[tp]
            parameter_val -= gamma * (state.uu_tp[(tp[0] + 1, tp[1])])

            # Calculates impact of transition in difficulties
            for mc in itertools.product(indices['m'],indices['c']):
                for d in range(len(indices['d'])):
                    if mc[0] == 0: transition_prob = 0
                    else: transition_prob = transition[(mc[0], indices['d'][d], mc[1])]
                    ps_val = state.ps_tmdc[(tp[0], mc[0], indices['d'][d], mc[1])]
                    if d == len(indices['d'])-1: usage_val = 0
                    else: usage_val = usage[(tp[1], indices['d'][d+1], mc[1])] - usage[(tp[1], indices['d'][d], mc[1])]

                parameter_val -= gamma * ps_val * transition_prob * usage_val


            # Calculates impact of scheduling
            for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
                parameter_val -= action.sc_tmdc[(tp[0], mdc[0], mdc[1], mdc[2])] * usage[(tp[1], mdc[1], mdc[2])]

            # Calcualtes impact of rescheduling into it
            for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
                parameter_val -= action.rsc_ttpmdc[(tmdc[0], tp[0]+1, tmdc[1], tmdc[2], tmdc[3])] * usage[(tp[1], tmdc[2], tmdc[3])]

            # Calcualtes impact of rescheduling out of it
            for tpmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
                parameter_val += action.rsc_ttpmdc[(tp[0]+1, tpmdc[0], tpmdc[1], tpmdc[2], tpmdc[3])] * usage[(tp[1], tpmdc[2], tpmdc[3])]
            
            
            lhs[tp] = parameter_val
        
        rhs[tp] = E_uu[tp]
        sign[tp] = ">="
        name[tp] = f"beta_uu_tp_{tp[0]}_{tp[1].replace(' ', '-')}"
    
    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data

# Generates beta pe constraint data
def b_pe_constraint(state: state, action: action) -> constraint_parameter:
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}
    
    # Creates Data
    for dc in itertools.product(indices['d'], indices['c']):
        lhs[dc] = state.pe_dc[dc] - gamma*(arrival[dc])
        rhs[dc] = E_pe[dc]
        sign[dc] = ">="
        name[dc] = f"beta_pe_dc_{dc[0].replace(' ','-')}_{dc[1].replace(' ', '-')}"
    
    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data

# Generates beta pw constraint data
def b_pw_constraint(state: state, action: action) -> constraint_parameter:
    pass

# Generates beta ps constraint data
def b_pw_constraint(state: state, action: action) -> constraint_parameter:
    pass

# Generates Gurobi Constraints from 
# A list of state-action-data & corresponding constraint
# And a function that generates constraint parameters for this
def generate_constraint( 
    model: gp.Model,
    state_action_data: List[ Tuple[state,action]], 
    state_action_variable: List[gp.Var],
    generator_function: Callable[ [state,action], constraint_parameter],
) -> Dict[ Tuple[str],gp.Constr ]:
    
    # Initialization
    expr: Dict[ Tuple[str],gp.LinExpr ] = {}
    init_iterables = generator_function(state_action_data[0][0], state_action_data[0][1])
    for index_iter in init_iterables.lhs_param.keys():
        expr[index_iter] = gp.LinExpr()
    
    # Generation
    for sa_pair in range(len(state_action_data)):
        params = generator_function(state_action_data[sa_pair][0], state_action_data[sa_pair][1])
        for index_iter in params.lhs_param.keys():
            expr[index_iter].add(state_action_variable[sa_pair], params.lhs_param[index_iter])

    # Saving
    results = {}
    for index_iter in init_iterables.lhs_param.keys():
        if init_iterables.sign[index_iter] == "=":
            results[index_iter] = model.addConstr(expr[index_iter] == init_iterables.rhs_param[index_iter], name=init_iterables.name[index_iter])
        elif init_iterables.sign[index_iter] == ">=":
            results[index_iter] = model.addConstr(expr[index_iter] >= init_iterables.rhs_param[index_iter], name=init_iterables.name[index_iter])
        elif init_iterables.sign[index_iter] == "<=":
            results[index_iter] = model.addConstr(expr[index_iter] <= init_iterables.rhs_param[index_iter], name=init_iterables.name[index_iter])
        else: 
            print('\terror')
    return(results)


# %% Master Model

w_sa: List[ Tuple[state,action] ] = [] # state-action pair list

w_sa.append((test_state, test_action))
w_sa.append((test_state, test_action))

mast_model = gp.Model('MasterKnapsack')

w_sa_var = []
for sa_pair in range(len(w_sa)):
    w_sa_var.append(mast_model.addVar(name=f'sa_{sa_pair}'))

# Constraints
generate_constraint(mast_model, w_sa, w_sa_var, b_0_constraint)
generate_constraint(mast_model, w_sa, w_sa_var, b_uu_constraint)
generate_constraint(mast_model, w_sa, w_sa_var, b_ue_constraint)
generate_constraint(mast_model, w_sa, w_sa_var, b_pe_constraint)

# Objective Function
    # Initialization
obj_expr = gp.LinExpr()
    # Generation
for sa_pair in range(len(w_sa)):
    cost_param = cost_function(w_sa[sa_pair][0], w_sa[sa_pair][1])
    obj_expr.add(w_sa_var[sa_pair], cost_param)
    # Saving
mast_model.setObjective(obj_expr, GRB.MINIMIZE)

mast_model.write('mast.lp')
mast_model.optimize()

for i in w_sa_var:
    print(i.x)

# %%
