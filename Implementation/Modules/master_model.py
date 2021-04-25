from dataclasses import dataclass
from Modules.data_classes import state, action, constraint_parameter, input_data_class
from typing import Dict, Tuple, List, Callable

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Initialization 
def generate_initial_state_action(input_data):
    # Input Data
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data

    # Patient States
    # pe = {}
    # for dc in itertools.product(indices['d'], indices['c']):
    #     pe[dc] = arrival[dc] * 4
    pw = {}
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        # if mdc[0] == 0: continue
        pw[mdc] = arrival[(mdc[1], mdc[2])] * 4
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        # if tmdc[1] == 0: continue
        ps[tmdc] = 0

    # Unit States
    ue = {}
    uu = {}
    uv = {}
    for tp in itertools.product(indices['t'], indices['p']):
        ub = ppe_data[tp[1]].expected_units + ppe_data[tp[1]].deviation[1] 
        ue[tp] = ub
        uv[tp] = ub
        if tp[0] == 1: uu[tp] = ub*2
        else: uu[tp] = 0

    # Actions
    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = 0
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        # if ttpmdc[2] == 0: continue
        rsc[ttpmdc] = 0

    # Returns
    test_state = state(ue, uu, uv, pw, ps)
    test_action = action(sc, rsc)
    return test_state, test_action

# Misc Functions to generate constrain parameters
# Cost Function
def cost_function(input_data: input_data_class, state: state, action: action):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    cost = 0

    # Cost of Waiting
    cost_waiting_unsc = 0
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):            
        cost += model_param.cw**mdc[0] * ( state.pw_mdc[mdc] )
    cost_waiting_unsc = cost

    # Cost of Waiting - Last Period
    cost_waiting_last = 0
    for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        cost += model_param.cw**indices['m'][-1] * ( state.ps_tmdc[(tdc[0],indices['m'][-1],tdc[1],tdc[2])] )
    cost_waiting_last = cost - cost_waiting_unsc

    # Cost of Cancelling
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] > ttpmdc[1]: #good schedule
            cost -= model_param.cc * action.rsc_ttpmdc[ttpmdc]
        elif ttpmdc[1] > ttpmdc[0]: #bad schedule
            cost += model_param.cc * action.rsc_ttpmdc[ttpmdc]

    # Violating unit bounds
    for tp in itertools.product(indices['t'], indices['p']):
        cost += state.uv_tp[tp] * M

    return(cost)
# Generates beta 0 constraint data 
def b_0_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    constr_data = constraint_parameter(
        {"b_0": 1-gamma}, 
        {"b_0": 1},
        {"b_0": '='},
        {"b_0": 'b_0'},
    )
    return constr_data
# Generates beta ue constraint data
def b_ue_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            previous_numbers = state.ue_tp[tp] - state.uu_tp[tp]
            change_due_to_schedule = 0
            for dc in itertools.product(indices['d'], indices['c']):
                for m in itertools.product(indices['m']):
                    change_due_to_schedule += usage[(tp[1], dc[0], dc[1])] * action.sc_tmdc[(tp[0],m[0], dc[0], dc[1])]
                for tpm in itertools.product(indices['t'], indices['m']):
                    change_due_to_schedule += usage[(tp[1], dc[0], dc[1])] * action.rsc_ttpmdc[(tp[0],tpm[0], tpm[1], dc[0], dc[1])]
                for tm in itertools.product(indices['t'], indices['m']):
                    change_due_to_schedule += usage[(tp[1], dc[0], dc[1])] * action.rsc_ttpmdc[(tm[0],tp[0], tm[1], dc[0], dc[1])]
            lhs[tp] = state.ue_tp[tp] - gamma*(ppe_data[tp[1]].expected_units + previous_numbers - change_due_to_schedule)

        elif tp[0] >= 2:
            lhs[tp] = state.ue_tp[tp] - gamma*(ppe_data[tp[1]].expected_units)

        rhs[tp] = input_data.expected_state_values['ue'][tp]
        sign[tp] = ">="
        name[tp] = f'b_ue_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}'

    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta uu constraint data
def b_uu_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):

        # When t is T
        if tp[0] == indices['t'][-1]:
            lhs[tp] = state.uu_tp[tp]
        
        # All others
        else:
            parameter_val = state.uu_tp[tp]
            parameter_val -= gamma * (state.uu_tp[(tp[0] + 1, tp[1])])

            # Calculates impact of transition in difficulties
            for mc in itertools.product(indices['m'],indices['c']):
                for d in range(len(indices['d'])):
                    
                    transition_prob = transition[(mc[0], indices['d'][d], mc[1])]
                    ps_val = state.ps_tmdc[(tp[0], mc[0], indices['d'][d], mc[1])]
                    if d == len(indices['d'])-1: usage_change = 0
                    else: usage_change = usage[(tp[1], indices['d'][d+1], mc[1])] - usage[(tp[1], indices['d'][d], mc[1])]

                    parameter_val -= gamma * ps_val * transition_prob * usage_change

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
        
        rhs[tp] = input_data.expected_state_values['uu'][tp]
        sign[tp] = ">="
        name[tp] = f'b_uu_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}'
    
    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta uv constraint data
def b_uv_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):
        lhs[tp] = state.uv_tp[tp]
        rhs[tp] = input_data.expected_state_values['uv'][tp]
        sign[tp] = ">="
        name[tp] = f'b_uv_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}'

    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta pw constraint data
def b_pw_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}
    
    # Creates Data
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            mdc = (mc[0], indices['d'][d], mc[1])

            # When m = 0
            if mc[0] == 0: 
                curr_state = state.pw_mdc[mdc]
                lhs[mdc] = curr_state - (gamma * arrival[(indices['d'][d], mc[1])] )

            # When m = M
            elif mc[0] == indices['m'][-1]:
                curr_state = state.pw_mdc[mdc]
                not_scheduled = 0
                transitioned_out = 0
                transitioned_in = 0

                for mm in input_data.indices['m'][-2:]:
                    not_scheduled += gamma * state.pw_mdc[(mm, indices['d'][d], mc[1])]
                    for t in indices['t']:
                        not_scheduled -= gamma * action.sc_tmdc[(t, mm, indices['d'][d], mc[1])]
                    transitioned_out = gamma * state.pw_mdc[(mm,indices['d'][d],mc[1])] * transition[(mm,indices['d'][d],mc[1])]
                    if d == 0: 
                        transitioned_in = 0
                    else:
                        transitioned_in = gamma * state.pw_mdc[(mm,indices['d'][d-1],mc[1])] * transition[(mm,indices['d'][d-1],mc[1])]
                        
                lhs[mdc] = curr_state - not_scheduled + transitioned_out - transitioned_in

            # All others
            else:                   
                curr_state = state.pw_mdc[mdc]
                not_scheduled = gamma * state.pw_mdc[(mc[0]-1, indices['d'][d], mc[1])]
                for t in indices['t']:
                    not_scheduled -= gamma * action.sc_tmdc[(t, mc[0]-1, indices['d'][d], mc[1])]
                transitioned_out = gamma * state.pw_mdc[(mc[0]-1,indices['d'][d],mc[1])] * transition[(mc[0]-1,indices['d'][d],mc[1])]
                if d == 0: 
                    transitioned_in = 0
                else:
                    transitioned_in = gamma * state.pw_mdc[(mc[0]-1,indices['d'][d-1],mc[1])] * transition[(mc[0]-1,indices['d'][d-1],mc[1])]
                lhs[mdc] = curr_state - not_scheduled + transitioned_out - transitioned_in

            rhs[mdc] = input_data.expected_state_values['pw'][mdc]
            sign[mdc] = ">="
            name[mdc] = f'b_pw_{str(mdc)[1:][:-1].replace(" ","_").replace(",","")}'
        
    # Returns Data
    constr_data = constraint_parameter(lhs,rhs,sign,name)
    return constr_data
# Generates beta ps constraint data
def b_ps_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}
    
    for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])

            # When m is 0 or t is T
            if (tmdc[1] == 0) or (tmdc[0] == indices['t'][-1]): 
                lhs[tmdc] = state.ps_tmdc[tmdc]

            # When m in (1...M-1) and t in (1..T-1)
            elif tmdc[1] != indices['m'][-1]:
                # Baseline
                lhs[tmdc] = state.ps_tmdc[tmdc]
                # Transition in difficulties
                if tmdc[1] >= 1:
                    lhs[tmdc] -= gamma * (1 - transition[( tmdc[1]-1, tmdc[2], tmdc[3] )]) * state.ps_tmdc[( tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3] )] 
                    if d != 0: 
                        lhs[tmdc] -= gamma * (transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )]) * state.ps_tmdc[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )] 
                
                # Scheduling / Rescheduling
                lhs[tmdc] += gamma * action.sc_tmdc[ (tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3]) ]
                if tmdc[1] >= 1:
                    for t in indices['t']:
                        lhs[tmdc] += gamma * action.rsc_ttpmdc[ (tmdc[0]+1, t, tmdc[1]-1, tmdc[2], tmdc[3]) ]
                        lhs[tmdc] -= gamma * action.rsc_ttpmdc[ (t, tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3]) ]
    
            # When m is M, t in (1...T-1)
            else:
                # Baseline
                lhs[tmdc] = state.ps_tmdc[tmdc]
                # Transition in difficulties
                for mm in indices['m'][-2:]:
                    lhs[tmdc] -= gamma * (1 - transition[( mm, tmdc[2], tmdc[3] )]) * state.ps_tmdc[( tmdc[0]+1, mm, tmdc[2], tmdc[3] )] 
                    if d != 0: 
                        lhs[tmdc] -= gamma * (transition[( mm, indices['d'][d-1], tmdc[3] )]) * state.ps_tmdc[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )] 
                    
                    # Scheduling / Rescheduling
                    lhs[tmdc] += gamma * action.sc_tmdc[ (tmdc[0]+1, mm, tmdc[2], tmdc[3] ) ]
                    for t in indices['t']:
                        lhs[tmdc] += gamma * action.rsc_ttpmdc[ (tmdc[0]+1, t, mm, tmdc[2], tmdc[3]) ]
                        lhs[tmdc] -= gamma * action.rsc_ttpmdc[ (t, tmdc[0]+1, mm, tmdc[2], tmdc[3]) ]
                
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

    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage
    
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
    gv_ue = {}
    gv_uu = {}
    gv_uv = {}
    gv_pw = {}
    gv_ps = {}

    # Constraints 
    b0_constr = {}
    ue_constr = {}
    uu_constr = {}
    uv_constr = {}
    pw_constr = {}
    ps_constr = {}

    # Adds variables to model, updates constraints, and adds to objective function
        # Beta 0
    gv_0['b_0']= mast_model.addVar(name=f'gv_b_0', obj=1)
    b0_constr['b_0'] = mast_model.getConstrByName('b_0')
    mast_model.chgCoeff(b0_constr['b_0'], gv_0['b_0'], 1)

    for tp in itertools.product(indices['t'], indices['p']):
        # Beta ue
        ue_constr[tp] = mast_model.getConstrByName(f'b_ue_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_ue[tp] = mast_model.addVar(name=f'gv_b_ue_{tp}', obj=1)
        mast_model.chgCoeff(ue_constr[tp], gv_ue[tp], 1)
        
        # Beta uu
        uu_constr[tp] = mast_model.getConstrByName(f'b_uu_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_uu[tp] = mast_model.addVar(name=f'gv_b_uu_{tp}', obj=1)
        mast_model.chgCoeff(uu_constr[tp], gv_uu[tp], 1)
        
        # Beta uv
        uv_constr[tp] = mast_model.getConstrByName(f'b_uv_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}')
        gv_uv[tp] = mast_model.addVar(name=f'gv_b_uv__{tp}', obj=1)
        mast_model.chgCoeff(uv_constr[tp], gv_uv[tp], 1)
    
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
        'ue': ue_constr,
        'uu': uu_constr,
        'uv': uv_constr,
        'pw': pw_constr,
        'ps': ps_constr
    }

    mast_model.update()
    return mast_model, constraints
# Generates Master model (finding optimal solution)
def generate_master_model(input_data, state_action_data):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
   
    # Model
    mast_model = gp.Model('MasterKnapsack')

    # Generates Variables
    w_sa_var = []
    for sa_pair in range(len(state_action_data)):
        w_sa_var.append(mast_model.addVar(name=f'sa_{sa_pair}'))

    # Generates Constraints
    b0_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_0_constraint)
    ue_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ue_constraint)
    uu_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_uu_constraint)
    uv_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_uv_constraint)
    pw_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_pw_constraint)
    ps_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ps_constraint)
    constraints = {
        'b0': b0_constr,
        'ue': ue_constr,
        'uu': uu_constr,
        'uv': uv_constr,
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
def update_master_model(input_data, master_model, master_variables, master_constraints, new_state_action, sa_index):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # Adds Variables Variables and updates objective function
    cost_val = cost_function(input_data, new_state_action[0], new_state_action[1])
    master_variables.append(master_model.addVar(name=f'sa_{sa_index}', obj=cost_val))

    # Modifies Constrain Values
    master_model.chgCoeff(master_constraints['b0']['b_0'], master_variables[-1], b_0_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param['b_0'])
    for tp in itertools.product(indices['t'], indices['p']):
        # Beta ue
        master_model.chgCoeff(master_constraints['ue'][tp], master_variables[-1], b_ue_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param[tp])
        # Beta uu
        master_model.chgCoeff(master_constraints['uu'][tp], master_variables[-1], b_uu_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param[tp])
        # Beta uv
        master_model.chgCoeff(master_constraints['uv'][tp], master_variables[-1], b_uv_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param[tp])

    # Beta pw
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        master_model.chgCoeff(master_constraints['pw'][mdc], master_variables[-1], b_pw_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param[mdc])
    
    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        master_model.chgCoeff(master_constraints['ps'][tmdc], master_variables[-1], b_ps_constraint(input_data, new_state_action[0], new_state_action[1]).lhs_param[tmdc])

    master_model.update()
    return master_model, master_variables, master_constraints

# Generates Beta Parameters
def generate_beta_values(input_data, constraints):
    indices = input_data.indices

    # Beta Values
    b_0_dual = {}
    b_ue_dual = {}
    b_uu_dual = {}
    b_uv_dual = {}
    b_pw_dual = {}
    b_ps_dual = {}

    # Beta 0
    b_0_dual['b_0'] = constraints['b0']['b_0'].Pi

    for tp in itertools.product(indices['t'], indices['p']):
        # Beta ue
        b_ue_dual[tp] = constraints['ue'][tp].Pi
        
        # Beta uu
        b_uu_dual[tp] = constraints['uu'][tp].Pi
        
        # Beta uv
        b_uv_dual[tp] = constraints['uv'][tp].Pi

    # Beta pw
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        b_pw_dual[mdc] = constraints['pw'][mdc].Pi

    # Beta ps
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        b_ps_dual[tmdc] = constraints['ps'][tmdc].Pi

    # Combines beta values
    betas = {
        'b0': b_0_dual,
        'ue': b_ue_dual,
        'uu': b_uu_dual,
        'uv': b_uv_dual,
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas