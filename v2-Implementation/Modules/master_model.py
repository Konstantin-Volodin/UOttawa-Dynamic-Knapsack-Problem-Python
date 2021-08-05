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
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        pw[mdc] = 0
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps[tmdc] = 0

    # Unit States
    ul = {}
    for p in itertools.product(indices['p']):
        ul[p] = 0

    # Actions
    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = 0
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = 0
    
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
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        pw_p[mdc] = 0
    ps_p = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_p[tmdc] = 0

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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):            
        cost += model_param.cw * ( action.pw_p_mdc[mdc] )
    
    # Cost of Waiting - Last Period
    # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
    #     cost += model_param.cw**(indices['m'][-1]+1) * ( action.ps_p_tmdc[(tdc[0],indices['m'][-1],tdc[1],tdc[2])] )

    # Prefer Earlier Appointments
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        cost += model_param.cs[tmdc[0]] * ( action.sc_tmdc[tmdc] )

    # Cost of Rescheduling
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] > ttpmdc[1]: # good schedule
            difference = ttpmdc[0] - ttpmdc[1]
            cost -= (model_param.cs[difference] - model_param.cc) * action.rsc_ttpmdc[ttpmdc]
        elif ttpmdc[1] > ttpmdc[0]: #bad schedule
            difference = ttpmdc[1] - ttpmdc[0]
            cost += (model_param.cs[difference] + model_param.cc) * action.rsc_ttpmdc[ttpmdc]

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
        
        if ppe_data[p[0]].ppe_type == 'carry-over':
            lhs[p] = state.ul_p[p] - gamma * action.ul_p_p[p]    
        elif ppe_data[p[0]].ppe_type == 'non-carry-over':
            lhs[p] = state.ul_p[p]

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
            mdc = (mc[0], indices['d'][d], mc[1])
            dc = (indices['d'][d], mc[1])

            # When m = 0
            if mc[0] == 0: 
                lhs[mdc] = state.pw_mdc[mdc] - (gamma * arrival[(mdc[1], mc[1])] )

            # When m = M
            elif mc[0] == indices['m'][-1]:
                lhs[mdc] = state.pw_mdc[mdc]

                for mm in input_data.indices['m'][-2:]:
                    lhs[mdc] -= gamma * action.pw_p_mdc[(mm, mdc[1], mdc[2])]
                    
                    transitioned_in = 0
                    if d != 0 & (mm >= transition[dc].wait_limit+1):
                        transitioned_in = transition[dc].transition_rate * action.pw_p_mdc[( mm, indices['d'][d-1], mdc[2] )]
                    transitioned_out = 0
                    if d != indices['d'][-1]:
                        transitioned_out = transition[dc].transition_rate * action.pw_p_mdc[( mm, mdc[1], mdc[2] )]
                    
                    lhs[mdc] -= gamma * (transitioned_in - transitioned_out )

            # When m is less than TL_dc
            elif mc[0] <= (transition[dc].wait_limit - 1):
                lhs[mdc] = state.pw_mdc[mdc] - gamma * action.pw_p_mdc[(mdc[0]-1,mdc[1],mdc[2])]

            # All others
            else:      
                lhs[mdc] = state.pw_mdc[mdc] - gamma*(action.pw_p_mdc[(mdc[0]-1,mdc[1],mdc[2])])

                transitioned_in = 0
                if d != 0 & (mdc[0] >= transition[dc].wait_limit+1):
                    transitioned_in = transition[dc].transition_rate * action.pw_p_mdc[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                transitioned_out = 0
                if d != indices['d'][-1]:
                    transitioned_out = transition[dc].transition_rate * action.pw_p_mdc[( mdc[0]-1, mdc[1], mdc[2] )]
                
                lhs[mdc] -= gamma * (transitioned_in - transitioned_out )

            rhs[mdc] = input_data.expected_state_values['pw'][mdc]
            sign[mdc] = ">="
            name[mdc] = f'b_pw_{str(mdc)[1:][:-1].replace(" ","_").replace(",","")}'
        
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
    
    for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])
            dc = (indices['d'][d], tmc[2])

            # When m is 0
            if tmdc[1] == 0: 
                lhs[tmdc] = state.ps_tmdc[tmdc]

            # When t is T
            elif tmdc[0] == indices['t'][-1]:
                lhs[tmdc] = state.ps_tmdc[tmdc]

            # when m is M
            elif tmdc[1] == indices['m'][-1]:
                lhs[tmdc] = state.ps_tmdc[tmdc]

                for mm in input_data.indices['m'][-2:]:
                    lhs[tmdc] -= gamma * action.ps_p_tmdc[(tmdc[0] + 1, mm, tmdc[2], tmdc[3])]

                    transitioned_in = 0
                    if d != 0 & (mm >= transition[dc].wait_limit+1):
                        transitioned_in = transition[dc].transition_rate * action.ps_p_tmdc[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                    transitioned_out = 0
                    if d != indices['d'][-1]:
                        transitioned_out = transition[dc].transition_rate * action.ps_p_tmdc[( tmdc[0]+1, mm, tmdc[2], tmdc[3] )]
                        
                    lhs[tmdc] -= gamma * (transitioned_in - transitioned_out )
                    
            # When m is less than TL_dc
            elif tmdc[1] <= (transition[dc].wait_limit - 1):
                lhs[tmdc] = state.ps_tmdc[tmdc] - gamma*(action.ps_p_tmdc[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
            
            # Everything Else
            else:
                lhs[tmdc] = state.ps_tmdc[tmdc] - gamma*(action.ps_p_tmdc[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
                
                transitioned_in = 0
                if d != 0  & (tmdc[1] >= transition[dc].wait_limit+1):
                    transitioned_in = transition[dc].transition_rate * action.ps_p_tmdc[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                transitioned_out = 0
                if d != indices['d'][-1]:
                    transitioned_out = transition[dc].transition_rate * action.ps_p_tmdc[( tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3] )]

                lhs[tmdc] -= gamma * (transitioned_in - transitioned_out )
                
            rhs[tmdc] = input_data.expected_state_values['ps'][tmdc]
            sign[tmdc] = ">="
            name[tmdc] = f'b_ps_{str(tmdc)[1:][:-1].replace(" ","_").replace(",","")}'

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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        pw_constr[mdc] = mast_model.getConstrByName(f'b_pw_{str(mdc)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_pw[mdc] = mast_model.addVar(name=f'gv_b_pw_{mdc}', obj=1)
        mast_model.chgCoeff(pw_constr[mdc], gv_pw[mdc], 1)
    
    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_constr[tmdc] = mast_model.getConstrByName(f'b_ps_{str(tmdc)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_ps[tmdc] = mast_model.addVar(name=f'gv_b_ps_{tmdc}', obj=1)
        mast_model.chgCoeff(ps_constr[tmdc], gv_ps[tmdc], 1)

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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        master_model.chgCoeff(master_constraints['pw'][mdc], master_variables[-1], pw_constraint_params.lhs_param[mdc])
 
    # Beta ps
    ps_constraint_params = b_ps_constraint(input_data, new_state_action[0], new_state_action[1])
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        master_model.chgCoeff(master_constraints['ps'][tmdc], master_variables[-1], ps_constraint_params.lhs_param[tmdc])
 
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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        b_pw_dual[mdc] = constraints['pw'][mdc].Pi

    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        b_ps_dual[tmdc] = constraints['ps'][tmdc].Pi

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ul': b_ul_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas