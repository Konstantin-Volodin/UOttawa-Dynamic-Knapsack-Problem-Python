from dataclasses import dataclass
from Modules.data_classes import state, action, constraint_parameter, input_data_class
from typing import Dict, Tuple, List, Callable

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import Modules.decorators
import time

# Initialization 
def generate_initial_state_action(input_data):
    # Input Data
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data

    # Patient States
    pw = {}
    for mdkc in itertools.product(indices['m'],indices['d'], indices['k'], indices['c']):
        pw[mdkc] = 0
    ps = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps[tmdkc] = 0

    # Unit States
    ul = {}
    for p in itertools.product(indices['p']):
        ul[p] = 0

    # Actions
    sc = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        sc[tmdkc] = 0
    rsc = {}
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        rsc[ttpmdkc] = 0
    
    # Violation
    uv = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uv[tp] = 0
    
    # Auxiliary variables
    ul_p = {}
    ulb = {}
    for p in itertools.product(indices['p']):
        ul_p[p] = 0
        ulb[p] = 0
    uvb = {}
    uu_p = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uvb[tp] = 0
        uu_p[tp] = 0
    pw_p = {}
    for mdkc in itertools.product(indices['m'],indices['d'], indices['k'], indices['c']):
        pw_p[mdkc] = 0
    ps_p = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps_p[tmdkc] = 0

    # Returns
    test_state = state(ul, pw, ps)
    test_action = action(sc, rsc, uv, uvb, ul_p, ulb, uu_p, pw_p, ps_p)
    return test_state, test_action

# Misc Functions to generate constrain parameters
# Cost Function
def cost_function(input_data: input_data_class, state: state, action: action):
    indices = input_data.indices
    model_param = input_data.model_param

    cost = 0
    # Cost of Waiting
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):            
        cost += model_param.cw[mdkc[2]] * ( action.pw_p_mdkc[mdkc] )

    # Prefer Earlier Appointments
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        cost += model_param.cs[tmdkc[3]][tmdkc[0]] * ( action.sc_tmdkc[tmdkc] )

    # Cost of Rescheduling
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        if ttpmdkc[0] > ttpmdkc[1]: # good schedule
            difference = ttpmdkc[0] - ttpmdkc[1]
            cost -= (model_param.cs[ttpmdkc[4]][difference] - model_param.cc[ttpmdkc[4]]) * action.rsc_ttpmdkc[ttpmdkc]
        elif ttpmdkc[1] > ttpmdkc[0]: #bad schedule
            difference = ttpmdkc[1] - ttpmdkc[0]
            cost += (model_param.cs[ttpmdkc[4]][difference] + model_param.cc[ttpmdkc[4]]) * action.rsc_ttpmdkc[ttpmdkc]

    # Violating unit bounds
    for tp in itertools.product(indices['t'], indices['p']):
        cost += action.uv_tp[tp] * model_param.M

    return cost
# Generates beta 0 constraint data 
def b_0_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    gamma = input_data.model_param.gamma

    constr_data = constraint_parameter(
        {"b_0": 1-gamma}, 
        {"b_0": 1},
        {"b_0": '='},
        {"b_0": 'b_0'},
    )
    return constr_data
# Generates beta ul constraint data
def b_ul_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    gamma = input_data.model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for p in itertools.product(indices['p']):

        # Left Hand Side        
        if ppe_data[p[0]].ppe_type == 'carry-over':
            lhs[p] = state.ul_p[p] - gamma * action.ul_p_p[p]    

        elif ppe_data[p[0]].ppe_type == 'non-carry-over':
            lhs[p] = state.ul_p[p]

        # Rest
        rhs[p] = input_data.expected_state_values['ul'][p]
        sign[p] = ">="
        name[p] = f'b_ul_{str(p)[1:][:-1].replace(" ","_").replace(",","")}'

    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta pw constraint data
def b_pw_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    transition = input_data.transition
    gamma = input_data.model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}
    
    # Creates Data
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):

                # Left Hand Side
                mdkc = (mc[0], indices['d'][d], indices['k'][k], mc[1])

                # When m = 0
                if mdkc[0] == 0: 
                    lhs[mdkc] = state.pw_mdkc[mdkc] - (gamma * arrival[(mdkc[1], mdkc[2], mdkc[3])] )

                # When m is less than TL_dc
                elif mdkc[0] < (transition.wait_limit[mdkc[3]]):
                    lhs[mdkc] = state.pw_mdkc[mdkc] - gamma * action.pw_p_mdkc[(mdkc[0]-1,mdkc[1],mdkc[2], mdkc[3])]

                # When m = M
                elif mdkc[0] == indices['m'][-1]:
                    lhs[mdkc] = state.pw_mdkc[mdkc]

                    for mm in input_data.indices['m'][-2:]:
                        lhs[mdkc] -= gamma * action.pw_p_mdkc[(mm, mdkc[1], mdkc[2], mdkc[3])]
                        
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[mdkc[3]]
                        tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                        
                        tr_in_d = 0
                        if (d != 0) & (mm >= tr_lim):
                            tr_in_d = tr_rate_d * action.pw_p_mdkc[( mm, indices['d'][d-1], mdkc[2], mdkc[3] )]
                            
                        tr_out_d = 0
                        if (d != indices['d'][-1]) & (mm >= tr_lim):
                            tr_out_d = tr_rate_d * action.pw_p_mdkc[( mm, mdkc[1], mdkc[2], mdkc[3] )]

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                        
                        tr_in_k = 0
                        if (k != 0) & (mm >= tr_lim):
                            tr_in_k = tr_rate_k * action.pw_p_mdkc[( mm, mdkc[1], indices['k'][k-1], mdkc[3] )]

                        tr_out_k = 0
                        if (k != indices['k'][-1]) & (mm >= tr_lim):
                            tr_out_k = tr_rate_k * action.pw_p_mdkc[( mm, mdkc[1], mdkc[2], mdkc[3] )]
                        
                        lhs[mdkc] -= gamma * (tr_in_d - tr_out_d )
                        lhs[mdkc] -= gamma * (tr_in_k - tr_out_k )


                # All others
                else:      
                    lhs[mdkc] = state.pw_mdkc[mdkc] - gamma*(action.pw_p_mdkc[(mdkc[0]-1,mdkc[1],mdkc[2],mdkc[3])])
                  
                    # Complexity Change
                    tr_lim = input_data.transition.wait_limit[mdkc[3]]
                    tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                    
                    tr_in_d = 0
                    if (d != 0) & (mdkc[0]-1 >= tr_lim):
                        tr_in_d = tr_rate_d * action.pw_p_mdkc[( mdkc[0]-1, indices['d'][d-1], mdkc[2], mdkc[3] )]
                            
                    tr_out_d = 0
                    if (d != indices['d'][-1]) & (mdkc[0]-1 >= tr_lim):
                        tr_out_d = tr_rate_d * action.pw_p_mdkc[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )]

                    # Priority Change
                    tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                    
                    tr_in_k = 0
                    if (k != 0) & (mdkc[0]-1 >= tr_lim):
                        tr_in_k = tr_rate_k * action.pw_p_mdkc[( mdkc[0]-1, mdkc[1], indices['k'][k-1], mdkc[3] )]

                    tr_out_k = 0
                    if (k != indices['k'][-1]) & (mdkc[0]-1 >= tr_lim):
                        tr_out_k = tr_rate_k * action.pw_p_mdkc[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )]
                    
                    lhs[mdkc] -= gamma * (tr_in_d - tr_out_d )
                    lhs[mdkc] -= gamma * (tr_in_k - tr_out_k )

                # Rest
                rhs[mdkc] = input_data.expected_state_values['pw'][mdkc]
                sign[mdkc] = ">="
                name[mdkc] = f'b_pw_{str(mdkc)[1:][:-1].replace(" ","_").replace(",","")}'
        
    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta ps constraint data
def b_ps_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    transition = input_data.transition
    gamma = input_data.model_param.gamma

    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}
    
    # Creates Data
    for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):

                # Left Hand Side
                tmdkc = (tmc[0], tmc[1], indices['d'][d], indices['k'][k], tmc[2])

                # When m = 0
                if tmdkc[1] == 0: 
                    lhs[tmdkc] = state.ps_tmdkc[tmdkc]

                # When t = T
                elif tmdkc[0] == indices['t'][-1]:
                    lhs[tmdkc] = state.ps_tmdkc[tmdkc]
                                    
                # When m is less than TL_c
                elif tmdkc[1] < (transition.wait_limit[tmdkc[4]]):
                    lhs[tmdkc] = state.ps_tmdkc[tmdkc] - gamma*(action.ps_p_tmdkc[(tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4])])
                
                # When m = M
                elif tmdkc[1] == indices['m'][-1]:
                    lhs[tmdkc] = state.ps_tmdkc[tmdkc]

                    for mm in input_data.indices['m'][-2:]:
                        lhs[tmdkc] -= gamma * action.ps_p_tmdkc[(tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4])]
                        
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                        tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                        
                        tr_in_d = 0
                        if (d != 0) & (mm >= tr_lim):
                            tr_in_d = tr_rate_d * action.ps_p_tmdkc[( tmdkc[0]+1, mm, indices['d'][d-1], tmdkc[3], tmdkc[4] )]
                            
                        tr_out_d = 0
                        if (d != indices['d'][-1]) & (mm >= tr_lim):
                            tr_out_d = tr_rate_d * action.ps_p_tmdkc[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )]

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                        
                        tr_in_k = 0
                        if (k != 0) & (mm >= tr_lim):
                            tr_in_k = tr_rate_k * action.ps_p_tmdkc[( tmdkc[0]+1, mm, tmdkc[2], indices['k'][k-1], tmdkc[4] )]

                        tr_out_k = 0
                        if (k != indices['k'][-1]) & (mm >= tr_lim):
                            tr_out_k = tr_rate_k * action.ps_p_tmdkc[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )]
                        
                        lhs[tmdkc] -= gamma * (tr_in_d - tr_out_d )
                        lhs[tmdkc] -= gamma * (tr_in_k - tr_out_k )

                # Everything Else
                else:
                    lhs[tmdkc] = state.ps_tmdkc[tmdkc] * action.ps_p_tmdkc[(tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4])]
                    
                    # Complexity Change
                    tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                    tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                    
                    tr_in_d = 0
                    if (d != 0) & (tmdkc[1]-1 >= tr_lim):
                        tr_in_d = tr_rate_d * action.ps_p_tmdkc[( tmdkc[0]+1, tmdkc[1]-1, indices['d'][d-1], tmdkc[3], tmdkc[4] )]
                        
                    tr_out_d = 0
                    if (d != indices['d'][-1]) & (tmdkc[1]-1 >= tr_lim):
                        tr_out_d = tr_rate_d * action.ps_p_tmdkc[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )]

                    # Priority Change
                    tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                    
                    tr_in_k = 0
                    if (k != 0) & (tmdkc[1]-1 >= tr_lim):
                        tr_in_k = tr_rate_k * action.ps_p_tmdkc[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], indices['k'][k-1], tmdkc[4] )]

                    tr_out_k = 0
                    if (k != indices['k'][-1]) & (tmdkc[1]-1 >= tr_lim):
                        tr_out_k = tr_rate_k * action.ps_p_tmdkc[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )]
                    
                    lhs[tmdkc] -= gamma * (tr_in_d - tr_out_d )
                    lhs[tmdkc] -= gamma * (tr_in_k - tr_out_k )
                
                # Rest
                rhs[tmdkc] = input_data.expected_state_values['ps'][tmdkc]
                sign[tmdkc] = ">="
                name[tmdkc] = f'b_ps_{str(tmdkc)[1:][:-1].replace(" ","_").replace(",","")}'

    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# generates constraints for a model
def generate_constraint( 
    model: gp.Model,
    input_data: input_data_class, 
    state_action_variable: List[gp.Var],
    state_action_data: List[ Tuple[state,action] ], 
    generator_function: Callable[ [state,action], constraint_parameter ],
) -> Dict[ Tuple[str],gp.Constr ]:

    # Initialization
    expr: Dict[ Tuple[str],gp.LinExpr ] = {}
    init_iterables = generator_function(input_data, state_action_data[0][0], state_action_data[0][1])
    for index_iter in init_iterables.lhs_param.keys():
        expr[index_iter] = gp.LinExpr()
    
    # Generation
    for sa_pair in range(len(state_action_data)):
        params = generator_function(input_data, state_action_data[sa_pair][0], state_action_data[sa_pair][1])
        for index_iter in params.lhs_param.keys():
            expr[index_iter].add(state_action_variable[sa_pair], round(params.lhs_param[index_iter],10))

    # Saving
    results = {}
    for index_iter in init_iterables.lhs_param.keys():
        if init_iterables.sign[index_iter] == "=":
            results[index_iter] = model.addConstr(expr[index_iter] == round(init_iterables.rhs_param[index_iter],10), name=init_iterables.name[index_iter])
        elif init_iterables.sign[index_iter] == ">=":
            results[index_iter] = model.addConstr(expr[index_iter] >= round(init_iterables.rhs_param[index_iter],10), name=init_iterables.name[index_iter])
        elif init_iterables.sign[index_iter] == "<=":
            results[index_iter] = model.addConstr(expr[index_iter] <= round(init_iterables.rhs_param[index_iter],10), name=init_iterables.name[index_iter])
        else: 
            print('\terror')
    return results
    

# Generate Phase 1 Master Model (finding initial feasible solution)
def generate_phase1_master_model(input_data, model):
    indices = input_data.indices    

    mast_model = model.copy()

    # Resets objective function
    obj_expr = gp.LinExpr()
    mast_model.setObjective(obj_expr, GRB.MINIMIZE)

    # Goal variables
    gv_0 = {}
    gv_ul = {}
    gv_pw = {}
    gv_ps = {}

    # Constraints 
    b0_constr = {}
    ul_constr = {}
    pw_constr = {}
    ps_constr = {}

    # Adds variables to model, updates constraints, and adds to objective function
        # Beta 0
    gv_0['b_0']= mast_model.addVar(name=f'gv_b_0', obj=1)
    b0_constr['b_0'] = mast_model.getConstrByName('b_0')
    mast_model.chgCoeff(b0_constr['b_0'], gv_0['b_0'], 1)

    for p in itertools.product(indices['p']):
        # Beta ul
        ul_constr[p] = mast_model.getConstrByName(f'b_ul_{str(p)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_ul[p] = mast_model.addVar(name=f'gv_b_ul_{p}', obj=1)
        mast_model.chgCoeff(ul_constr[p], gv_ul[p], 1)
        
    # Beta pw
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        pw_constr[mdkc] = mast_model.getConstrByName(f'b_pw_{str(mdkc)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_pw[mdkc] = mast_model.addVar(name=f'gv_b_pw_{mdkc}', obj=1)
        mast_model.chgCoeff(pw_constr[mdkc], gv_pw[mdkc], 1)
    
    # Beta ps
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps_constr[tmdkc] = mast_model.getConstrByName(f'b_ps_{str(tmdkc)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_ps[tmdkc] = mast_model.addVar(name=f'gv_b_ps_{tmdkc}', obj=1)
        mast_model.chgCoeff(ps_constr[tmdkc], gv_ps[tmdkc], 1)

    # Saves constraints
    constraints = {
        'b0': b0_constr,
        'ul': ul_constr,
        'pw': pw_constr,
        'ps': ps_constr
    }

    mast_model.update()
    return mast_model, constraints
# Generates Master model (finding optimal solution)
def generate_master_model(input_data, state_action_data):

    # Model
    mast_model = gp.Model('MasterKnapsack')

    # Generates Variables
    w_sa_var = []
    for sa_pair in range(len(state_action_data)):
        w_sa_var.append(mast_model.addVar(name=f'sa_{sa_pair}'))

    # Generates Constraints
    b0_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_0_constraint)
    ul_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ul_constraint)
    pw_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_pw_constraint)
    ps_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ps_constraint)
    constraints = {
        'b0': b0_constr,
        'ul': ul_constr,
        'pw': pw_constr,
        'ps': ps_constr
    }

    # Generates Objective Function
    obj_expr = gp.LinExpr()
    for sa_pair in range(len(state_action_data)):
        cost_param = cost_function(input_data, state_action_data[sa_pair][0], state_action_data[sa_pair][1])
        obj_expr.add(w_sa_var[sa_pair], cost_param)
    mast_model.setObjective(obj_expr, GRB.MINIMIZE)

    mast_model.update()

    # Returns model
    return mast_model, w_sa_var, constraints
# Updates master problem
def update_master_model(input_data, master_model, master_variables, master_constraints, new_state_action, sa_index):
    
    indices = input_data.indices

    # Adds Variables Variables and updates objective function
    cost_val = cost_function(input_data, new_state_action[0], new_state_action[1])


    master_variables.append(master_model.addVar(name=f'sa_{sa_index}', obj=cost_val))

    # Modifies Constrain Values
    # Beta b0
    b0_constraint_params = b_0_constraint(input_data, new_state_action[0], new_state_action[1])
    master_model.chgCoeff(master_constraints['b0']['b_0'], master_variables[-1], b0_constraint_params.lhs_param['b_0'])
    
    # Beta ul
    ul_constraint_params = b_ul_constraint(input_data, new_state_action[0], new_state_action[1])
    for p in itertools.product(indices['p']):
        master_model.chgCoeff(master_constraints['ul'][p], master_variables[-1], ul_constraint_params.lhs_param[p])
 
    # Beta pw
    pw_constraint_params = b_pw_constraint(input_data, new_state_action[0], new_state_action[1])
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        master_model.chgCoeff(master_constraints['pw'][mdkc], master_variables[-1], pw_constraint_params.lhs_param[mdkc])
 
    # Beta ps
    ps_constraint_params = b_ps_constraint(input_data, new_state_action[0], new_state_action[1])
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        master_model.chgCoeff(master_constraints['ps'][tmdkc], master_variables[-1], ps_constraint_params.lhs_param[tmdkc])
 
    master_model.update()
    return master_model, master_variables, master_constraints

# Generates Beta Parameters
def generate_beta_values(input_data, constraints):
    indices = input_data.indices

    # Beta Values
    b_0_dual = {}
    b_ul_dual = {}
    b_pw_dual = {}
    b_ps_dual = {}

    # Beta 0
    b_0_dual['b_0'] = constraints['b0']['b_0'].Pi

    for p in itertools.product(indices['p']):
        # Beta ul
        b_ul_dual[p] = constraints['ul'][p].Pi

    # Beta pw
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        b_pw_dual[mdkc] = constraints['pw'][mdkc].Pi

    # Beta ps
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        b_ps_dual[tmdkc] = constraints['ps'][tmdkc].Pi

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ul': b_ul_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas