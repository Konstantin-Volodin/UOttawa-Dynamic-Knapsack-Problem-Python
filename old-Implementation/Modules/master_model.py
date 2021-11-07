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
    pw = {}
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        pw[mdc] = 0
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps[tmdc] = 0

    # Unit States
    ue = {}
    uu = {}
    for tp in itertools.product(indices['t'], indices['p']):
        ue[tp] = ppe_data[tp[1]].expected_units
        uu[tp] = 0

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
    uu_p = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uu_p[tp] = uu[tp]
    pw_p = {}
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        pw_p[mdc] = 0
    ps_p = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_p[tmdc] = 0

    # Returns
    test_state = state(ue, uu, pw, ps)
    test_action = action(sc, rsc, uv, uu_p, pw_p, ps_p)
    return test_state, test_action

# Misc Functions to generate constrain parameters
# Cost Function
def cost_function(input_data: input_data_class, state: state, action: action):
    indices = input_data.indices
    model_param = input_data.model_param
    
    cost = 0
    # Cost of Waiting
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):            
        cost += model_param.cw**mdc[0] * ( action.pw_p_mdc[mdc] )
    
    # Cost of Waiting - Last Period
    for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        cost += model_param.cw**indices['m'][-1] * ( action.ps_p_tmdc[(tdc[0],indices['m'][-1],tdc[1],tdc[2])] )

    # Prefer Earlier Appointments
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        cost += model_param.cs**tmdc[0] * ( action.sc_tmdc[(tmdc[0],tmdc[1],tmdc[2],tmdc[3])] )

    # Cost of Rescheduling
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] > ttpmdc[1]: #good schedule
            cost -= (0.5*model_param.cc) * action.rsc_ttpmdc[ttpmdc]
        elif ttpmdc[1] > ttpmdc[0]: #bad schedule
            cost += (1.5*model_param.cc) * action.rsc_ttpmdc[ttpmdc]

    # Violating unit bounds
    for tp in itertools.product(indices['t'], indices['p']):
        cost += action.uv_tp[tp] * model_param.M

    return(cost)
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
# Generates beta ue constraint data
def b_ue_constraint(input_data: input_data_class, state: state, action: action) -> constraint_parameter:
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    gamma = input_data.model_param.gamma
    
    # Initializes Data
    lhs = {}
    rhs = {}
    sign = {}
    name = {}

    # Creates Data
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            lhs[tp] = state.ue_tp[tp] - gamma*( ppe_data[tp[1]].expected_units + state.ue_tp[tp] - action.uu_p_tp[tp] + action.uv_tp[tp])
       
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
    transition = input_data.transition
    usage = input_data.usage
    gamma = input_data.model_param.gamma
    
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
            parameter_val = state.uu_tp[tp] - gamma * (action.uu_p_tp[(tp[0] + 1, tp[1])])

            # Calculates impact of transition in difficulties
            for mc in itertools.product(indices['m'],indices['c']):
                for d in range(len(indices['d'])):

                    # When d is D
                    if d == len(indices['d'])-1: 
                        pass
                        
                    # Otherwise
                    else:
                        transition_prob = transition[(mc[0], indices['d'][d], mc[1])]
                        patients_sched = action.ps_p_tmdc[(tp[0]+1, mc[0], indices['d'][d], mc[1])]
                        expected_transition = transition_prob * patients_sched
                    
                        change_in_usage = usage[(tp[1], indices['d'][d+1], mc[1])] - usage[(tp[1], indices['d'][d], mc[1])]
                        parameter_val -= gamma * (expected_transition * change_in_usage)

            lhs[tp] = parameter_val
        
        rhs[tp] = input_data.expected_state_values['uu'][tp]
        sign[tp] = ">="
        name[tp] = f'b_uu_{str(tp)[1:][:-1].replace(" ","_").replace(",","")}'
    
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

            # When m = 0
            if mc[0] == 0: 
                lhs[mdc] = state.pw_mdc[mdc] - (gamma * arrival[(mdc[1], mc[1])] )

            # When m = M
            elif mc[0] == indices['m'][-1]:
                lhs[mdc] = state.pw_mdc[mdc]

                for mm in input_data.indices['m'][-2:]:
                    lhs[mdc] -= gamma * action.pw_p_mdc[(mm, mdc[1], mdc[2])]
                    
                    transitioned_in = 0
                    if d != 0:
                        transitioned_in = transition[( mm, indices['d'][d-1], mdc[2] )] * action.pw_p_mdc[( mm, indices['d'][d-1], mdc[2] )]
                    transitioned_out = 0
                    if d != indices['d'][-1]:
                        transitioned_out = transition[( mm, mdc[1], mdc[2] )] * action.pw_p_mdc[( mm, mdc[1], mdc[2] )]
                    
                    lhs[mdc] -= gamma * (transitioned_in - transitioned_out )

            # All others
            else:      
                lhs[mdc] = state.pw_mdc[mdc] - gamma*(action.pw_p_mdc[(mdc[0]-1,mdc[1],mdc[2])])

                transitioned_in = 0
                if d != 0:
                    transitioned_in = transition[( mdc[0]-1, indices['d'][d-1], mdc[2] )] * action.pw_p_mdc[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                transitioned_out = 0
                if d != indices['d'][-1]:
                    transitioned_out = transition[( mdc[0]-1, mdc[1], mdc[2] )] * action.pw_p_mdc[( mdc[0]-1, mdc[1], mdc[2] )]
                
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
                    if d != 0:
                        transitioned_in = transition[( mm, indices['d'][d-1], tmdc[3] )] * action.ps_p_tmdc[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                    transitioned_out = 0
                    if d != indices['d'][-1]:
                        transitioned_out = transition[( mm, tmdc[2], tmdc[3] )] * action.ps_p_tmdc[( tmdc[0]+1, mm, tmdc[2], tmdc[3] )]
                        
                    lhs[tmdc] -= gamma * (transitioned_in - transitioned_out )

            # Everything Else
            else:
                lhs[tmdc] = state.ps_tmdc[tmdc] - gamma*(action.ps_p_tmdc[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
                
                transitioned_in = 0
                if d != 0:
                    transitioned_in = transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )] * action.ps_p_tmdc[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                transitioned_out = 0
                if d != indices['d'][-1]:
                    transitioned_out = transition[( tmdc[1]-1, tmdc[2], tmdc[3] )] * action.ps_p_tmdc[( tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3] )]

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
    gv_ue = {}
    gv_uu = {}
    gv_pw = {}
    gv_ps = {}

    # Constraints 
    b0_constr = {}
    ue_constr = {}
    uu_constr = {}
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
    ue_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ue_constraint)
    uu_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_uu_constraint)
    pw_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_pw_constraint)
    ps_constr = generate_constraint(mast_model, input_data, w_sa_var, state_action_data, b_ps_constraint)
    constraints = {
        'b0': b0_constr,
        'ue': ue_constr,
        'uu': uu_constr,
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
    b_pw_dual = {}
    b_ps_dual = {}

    # Beta 0
    b_0_dual['b_0'] = constraints['b0']['b_0'].Pi

    for tp in itertools.product(indices['t'], indices['p']):
        # Beta ue
        b_ue_dual[tp] = constraints['ue'][tp].Pi
        
        # Beta uu
        b_uu_dual[tp] = constraints['uu'][tp].Pi

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
        'pw': b_pw_dual,
        'ps': b_ps_dual
    }

    return betas