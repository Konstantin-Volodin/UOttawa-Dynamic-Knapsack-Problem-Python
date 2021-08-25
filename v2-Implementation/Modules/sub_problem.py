from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from Modules.data_classes import input_data_class, state, action, variables

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB


    # Objective Function
    # Cost Function
def wait_cost(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    model_param = input_data.model_param
    expr = gp.LinExpr()

    # Cost of Waiting
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):  
        expr.addTerms(model_param.cw[mdkc[2]], var.a_pw_p[mdkc])                     

    return expr
def pref_earlier_appointment(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    model_param = input_data.model_param
    expr = gp.LinExpr()
    
    # Prefer Earlier Appointments
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr.addTerms(model_param.cs[tmdkc[3]][tmdkc[0]], var.a_sc[tmdkc])

    return expr
def reschedule_cost(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    model_param = input_data.model_param
    expr = gp.LinExpr()

    # Cost of Rescheduling                
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        if ttpmdkc[0] > ttpmdkc[1]: # Good Reschedule
            difference = ttpmdkc[0] - ttpmdkc[1]
            expr.addTerms(-(model_param.cs[ttpmdkc[4]][difference] - model_param.cc[ttpmdkc[4]]), var.a_rsc[ttpmdkc])
        elif ttpmdkc[0] < ttpmdkc[1]: # Bad Reschedule
            difference = ttpmdkc[1] - ttpmdkc[0]
            expr.addTerms((model_param.cs[ttpmdkc[4]][difference] - model_param.cc[ttpmdkc[4]]), var.a_rsc[ttpmdkc])

    return expr
def goal_violation_cost(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    model_param = input_data.model_param
    expr = gp.LinExpr()

    # Modification
    for tp in itertools.product(indices['t'], indices['p']):
        expr.addTerms(model_param.M, var.a_uv[tp])

    return expr

# E[V] Function
def b0_cost(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    model_param = input_data.model_param

    # Modification
    expr = gp.LinExpr()
    expr.addConstant( round((1-model_param.gamma) * betas['b0']['b_0'],10) )
    return expr
def b_ul_cost(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    gamma = input_data.model_param.gamma
    expr = gp.LinExpr()
    
    # Modification
    for p in itertools.product(indices['p']):   

        if ppe_data[p[0]].ppe_type == 'carry-over':
            expr.addTerms( round(betas['ul'][p],10), var.s_ul[p] )
            expr.addTerms(- round(betas['ul'][p] * gamma,10),  var.a_ul_p[p])  

        elif ppe_data[p[0]].ppe_type == 'non-carry-over':
            expr.addTerms( round(betas['ul'][p],10), var.s_ul[p])
                
    return expr
def b_pw_costs(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    arrival = input_data.arrival
    transition = input_data.transition
    gamma = input_data.model_param.gamma
    expr = gp.LinExpr()

    # Modification
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):

                mdkc = (mc[0], indices['d'][d], indices['k'][k], mc[1])

                # When m = 0
                if mdkc[0] == 0: 
                    # print(mdkc)
                    # print(betas['pw'][mdkc])
                    # print(var.s_pw[mdkc])
                    expr.addTerms( round(betas['pw'][mdkc],10), var.s_pw[mdkc] )
                    expr.addConstant(- round(betas['pw'][mdkc] * gamma * arrival[(mdkc[1], mdkc[2], mdkc[3])],10) )

                # When m is less than TL_c
                elif mdkc[0] < (transition.wait_limit[mdkc[3]]):
                    expr.addTerms( round(betas['pw'][mdkc],10), var.s_pw[mdkc])
                    expr.addTerms(- round((betas['pw'][mdkc] * gamma),10), var.a_pw_p[(mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3])] )

                # When m = M
                elif mdkc[0] == indices['m'][-1]:
                    expr.addTerms( round(betas['pw'][mdkc],10), var.s_pw[mdkc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(- round(betas['pw'][mdkc] * gamma,10), var.a_pw_p[(mm, mdkc[1], mdkc[2], mdkc[3])] )
        
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[mdkc[3]]
                        tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                        
                        if (d != 0) & (mm >= tr_lim):
                            expr.addTerms(- round(betas['pw'][mdkc] * gamma * tr_rate_d,10), var.a_pw_p[( mm, indices['d'][d-1], mdkc[2], mdkc[3] )] )
                            
                        if (d != indices['d'][-1]) & (mm >= tr_lim):
                            expr.addTerms( round(betas['pw'][mdkc] * gamma * tr_rate_d,10), var.a_pw_p[( mm, mdkc[1], mdkc[2], mdkc[3] )] )

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                        
                        if (k != 0) & (mm >= tr_lim):
                            expr.addTerms(- round(betas['pw'][mdkc] * gamma * tr_rate_k,10), var.a_pw_p[( mm, mdkc[1], indices['k'][k-1], mdkc[3] )] )

                        
                        if (k != indices['k'][-1]) & (mm >= tr_lim):
                            expr.addTerms( round(betas['pw'][mdkc] * gamma * tr_rate_k,10), var.a_pw_p[( mm, mdkc[1], mdkc[2], mdkc[3] )] )

                # Everything Else
                else:          
                    expr.addTerms( round(betas['pw'][mdkc],10), var.s_pw[mdkc] )
                    expr.addTerms(- round(betas['pw'][mdkc] * gamma,10), var.a_pw_p[(mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3])] )
        
                    # Complexity Change
                    tr_lim = input_data.transition.wait_limit[mdkc[3]]
                    tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                    
                    if (d != 0) & (mdkc[0]-1 >= tr_lim):
                        expr.addTerms(- round(betas['pw'][mdkc] * gamma * tr_rate_d,10), var.a_pw_p[( mdkc[0]-1, indices['d'][d-1], mdkc[2], mdkc[3] )] )
                        
                    if (d != indices['d'][-1]) & (mdkc[0]-1 >= tr_lim):
                        expr.addTerms( round(betas['pw'][mdkc] * gamma * tr_rate_d,10), var.a_pw_p[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )] )

                    # Priority Change
                    tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                    
                    if (k != 0) & (mdkc[0]-1 >= tr_lim):
                        expr.addTerms(- round(betas['pw'][mdkc] * gamma * tr_rate_k,10), var.a_pw_p[( mdkc[0]-1, mdkc[1], indices['k'][k-1], mdkc[3] )] )

                    
                    if (k != indices['k'][-1]) & (mdkc[0]-1 >= tr_lim):
                        expr.addTerms( round(betas['pw'][mdkc] * gamma * tr_rate_k,10), var.a_pw_p[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )] )

    return expr
def b_ps_costs(input_data:input_data_class, var: variables, betas) -> gp.LinExpr:
    # Initialization
    indices = input_data.indices
    arrival = input_data.arrival
    transition = input_data.transition
    gamma = input_data.model_param.gamma
    expr = gp.LinExpr()

    for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):
                
                tmdkc = (tmc[0], tmc[1], indices['d'][d], indices['k'][k], tmc[2])

                # When m = 0
                if tmdkc[0] == 0: 
                    expr.addTerms( round(betas['ps'][tmdkc],10), var.s_ps[tmdkc] )

                # When t = T
                elif tmdkc[0] == indices['t'][-1]:
                    expr.addTerms( round(betas['ps'][tmdkc],10), var.s_ps[tmdkc] )

                # When m is less than TL_c
                elif tmdkc[1] < (transition.wait_limit[tmdkc[4]]):
                    expr.addTerms( round(betas['ps'][tmdkc],10), var.s_ps[tmdkc] )
                    expr.addTerms(- round(betas['ps'][tmdkc] * gamma,10), var.s_ps[tmdkc] )

                # When m = M
                elif tmdkc[1] == indices['m'][-1]:
                    expr.addTerms( round(betas['ps'][tmdkc],10), var.s_ps[tmdkc] )

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(- round(betas['ps'][tmdkc] * gamma,10), var.s_ps[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )] )
        
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                        tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                        
                        if (d != 0) & (mm >= tr_lim):
                            expr.addTerms(- round(betas['ps'][tmdkc] * gamma * tr_rate_d,10), var.a_ps_p[( tmdkc[0]+1, mm, indices['d'][d-1], tmdkc[3], tmdkc[4] )] )
                            
                        if (d != indices['d'][-1]) & (mm >= tr_lim):
                            expr.addTerms( round(betas['ps'][tmdkc] * gamma * tr_rate_d,10), var.a_ps_p[ (tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4]) ] )

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                        
                        if (k != 0) & (mm >= tr_lim):
                            expr.addTerms(- round(betas['ps'][tmdkc] * gamma * tr_rate_k,10), var.a_ps_p[( tmdkc[0]+1, mm, tmdkc[2], indices['k'][k-1], tmdkc[4] )] )

                        
                        if (k != indices['k'][-1]) & (mm >= tr_lim):
                            expr.addTerms( round(betas['ps'][tmdkc] * gamma * tr_rate_k,10), var.a_ps_p[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )] )

                # Everything Else
                else:
                    expr.addTerms( round(betas['ps'][tmdkc],10), var.s_ps[tmdkc] )
                    expr.addTerms(- round(betas['ps'][tmdkc] * gamma,10), var.s_ps[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )] )
    
                    # Complexity Change
                    tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                    tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                    
                    if (d != 0) & (tmdkc[1]-1 >= tr_lim):
                        expr.addTerms(- round(betas['ps'][tmdkc] * gamma * tr_rate_d,10), var.a_ps_p[( tmdkc[0]+1, tmdkc[1]-1, indices['d'][d-1], tmdkc[3], tmdkc[4] )] )
                        
                    if (d != indices['d'][-1]) & (tmdkc[1]-1 >= tr_lim):
                        expr.addTerms( round(betas['ps'][tmdkc] * gamma * tr_rate_d,10), var.a_ps_p[ (tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4]) ] )

                    # Priority Change
                    tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                    
                    if (k != 0) & (tmdkc[1]-1 >= tr_lim):
                        expr.addTerms(- round(betas['ps'][tmdkc] * gamma * tr_rate_k,10), var.a_ps_p[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], indices['k'][k-1], tmdkc[4] )] )

                    
                    if (k != indices['k'][-1]) & (tmdkc[1]-1 >= tr_lim):
                        expr.addTerms( round(betas['ps'][tmdkc] * gamma * tr_rate_k,10), var.a_ps_p[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )] )     

    return expr
  

# Generate sub problem model
def generate_sub_model(input_data, betas, phase1 = False):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    usage = input_data.usage
    M = model_param.M

    # Initializes model
    sub_model = gp.Model('SubmodelKnapsack')

    # Decision Variables
    # State
    var_ul = {}
    var_pw = {}
    var_ps = {}

    # Action
    var_sc = {}
    var_rsc = {}
    var_uv = {}

    var_uvb = {}
    var_ul_p = {}
    var_ulb = {}
    var_uu_p = {}
    var_pw_p = {}
    var_ps_p = {}

    # States / Upper Bounds
    # UL
    for p in itertools.product(indices['p']):
        var_ul[p] = sub_model.addVar(name=f's_ul_{p}', ub=2*ppe_data[p[0]].expected_units, vtype=GRB.CONTINUOUS)
    # PW
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        var_pw[mdkc] = sub_model.addVar(name=f's_pw_{mdkc}', ub=2*arrival[(mdkc[1],mdkc[2],mdkc[3])], vtype=GRB.INTEGER)
    # PS
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_ps[tmdkc] = sub_model.addVar(name=f's_ps_{tmdkc}', ub=1*arrival[(tmdkc[2],tmdkc[3], tmdkc[4])], vtype=GRB.INTEGER)

    # Actions
    # SC
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_sc[tmdkc] = sub_model.addVar(name=f'a_sc_{tmdkc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_rsc[ttpmdkc] = sub_model.addVar(name=f'a_rsc_{ttpmdkc}', vtype=GRB.INTEGER)
    # UV
    for tp in itertools.product(indices['t'], indices['p']):
        var_uv[tp] = sub_model.addVar(name=f'a_uv_{tp}', vtype=GRB.CONTINUOUS)

    
    # UL Hat & UL B
    for p in itertools.product(indices['p']):
        if input_data.ppe_data[p[0]].ppe_type == 'carry-over':
            var_ul_p[p] = sub_model.addVar(name=f'a_ul_p_{p}', vtype=GRB.CONTINUOUS)
            var_ulb[p] = sub_model.addVar(name=f'a_ulb_{p}', vtype=GRB.BINARY)
    # UU Hat & UV B
    for tp in itertools.product(indices['t'], indices['p']):
        var_uu_p[tp] = sub_model.addVar(name=f'a_uu_p_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = sub_model.addVar(name=f'a_uvb_{tp}', vtype=GRB.BINARY)
    
    # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        var_pw_p[mdkc] = sub_model.addVar(name=f'a_pw_p_{mdkc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_ps_p[tmdkc] = sub_model.addVar(name=f'a_ps_p_{tmdkc}', vtype=GRB.INTEGER)

    sub_vars = variables(
        var_ul, var_pw, var_ps,
        var_sc, var_rsc, var_uv, 
        var_uvb, var_ul_p, var_ulb, var_uu_p, var_pw_p, var_ps_p
    )

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        expr.addTerms
        for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdkc[1], mdkc[3])], var_ps_p[(tp[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        sub_model.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdkc])
        expr.addTerms(-1, var_pw[mdkc])
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        sub_model.addConstr(expr == 0, name=f'pw_hat_{mdkc}')
        # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdkc])
        expr.addTerms(-1, var_ps[tmdkc])
        expr.addTerms(-1, var_sc[tmdkc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdkc[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
            expr.addTerms(1, var_rsc[(tmdkc[0], tp, tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        sub_model.addConstr(expr == 0, name=f'ps_hat_{tmdkc}')
        # UV Maximum function
    for tp in itertools.product(indices['t'], indices['p']):
        sub_model.addConstr(var_uv[tp] <= M * var_uvb[tp])
        
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        expr.addConstant(-input_data.ppe_data[tp[1]].expected_units)
        expr.addConstant(M)
        expr.addTerms(-M, var_uvb[tp])
        if tp[0] == 1:
            expr.addTerms(-1, var_ul[(tp[1],)])
        sub_model.addConstr(var_uv[tp] <= expr)
        # UL Maximum function
    for p in itertools.product(indices['p']):
        if input_data.ppe_data[p[0]].ppe_type == 'carry-over':
            sub_model.addConstr(var_ul_p[p] >= 0)
            sub_model.addConstr(var_ul_p[p] >= input_data.ppe_data[p[0]].expected_units + var_ul[p] - var_uu_p[(1, p[0])])
            sub_model.addConstr(var_ul_p[p] <= M * var_ulb[p])
            sub_model.addConstr(var_ul_p[p] <= input_data.ppe_data[p[0]].expected_units + var_ul[p] - var_uu_p[(1, p[0])] + (M * (1-var_ulb[p])))

    # Constraints
    # 1) Resource Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            # Carry Over
            if input_data.ppe_data[tp[1]].ppe_type == 'carry-over': 
                sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_ul[(tp[1],)] + var_uv[tp], name=f'resource_constraint_{tp}')
            # Non Carry Over
            elif input_data.ppe_data[tp[1]].ppe_type == 'non-carry-over':  
                sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')
        # Other Periods
        else:
            sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')

    # 2) Bounds on Reschedules
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        if ttpmdkc[0] == ttpmdkc[1] == 1:
            sub_model.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')
        elif ttpmdkc[0] >= 2 and ttpmdkc[1] >= 2:
            sub_model.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdkc[0], tp[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        expr.addTerms(1, var_ps[tmdkc])
        sub_model.addConstr(expr >= 0, f'consistent_resc_{(tmdkc)}')
        # Scheduled
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        expr.addTerms(1, var_pw[mdkc])
        sub_model.addConstr(expr >= 0, f'consistent_sch_{(mdkc)}')
  
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(input_data, sub_vars, betas)
    pref_early = pref_earlier_appointment(input_data, sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(input_data, sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(input_data, sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(input_data, sub_vars, betas)
    b_ul_expr = b_ul_cost(input_data, sub_vars, betas)
    b_pw_expr = b_pw_costs(input_data, sub_vars, betas)
    b_ps_expr = b_ps_costs(input_data, sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ul_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(-value_expr, GRB.MINIMIZE)
    else:
        sub_model.setObjective(cost_expr - value_expr, GRB.MINIMIZE)

    return sub_model, sub_vars

# Updates sub problem model
def update_sub_model(input_data, model, variables, betas, phase1 = False):
    sub_model = model
    sub_vars = variables

    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(input_data, sub_vars, betas)
    pref_early = pref_earlier_appointment(input_data, sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(input_data, sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(input_data, sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(input_data, sub_vars, betas)
    b_ul_expr = b_ul_cost(input_data, sub_vars, betas)
    b_pw_expr = b_pw_costs(input_data, sub_vars, betas)
    b_ps_expr = b_ps_costs(input_data, sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ul_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(-value_expr, GRB.MINIMIZE)
    else:
        sub_model.setObjective(cost_expr - value_expr, GRB.MINIMIZE)

    return sub_model, sub_vars

# Reads variables results and returns them
def generate_state_action(var: variables) -> Tuple[state, action]:
    
    # Extracts
    s_ul = {}
    for key, value in var.s_ul.items():
        s_ul[key] = value.x
        
    s_pw = {}
    for key, value in var.s_pw.items():
        s_pw[key] = int(value.x)
    
    s_ps = {}
    for key, value in var.s_ps.items():
        s_ps[key] = int(value.x)
    
    a_sc = {}
    for key, value in var.a_sc.items():
        a_sc[key] = int(value.x)

    a_rsc = {}
    for key, value in var.a_rsc.items():
        a_rsc[key] = int(value.x)

    a_uv = {}
    for key, value in var.a_uv.items():
        a_uv[key] = value.x

    a_uvb = {}
    for key, value in var.a_uvb.items():
        a_uvb[key] = value.x

    a_ul_p = {}
    for key, value in var.a_ul_p.items():
        a_ul_p[key] = value.x
        
    a_ulb = {}
    for key, value in var.a_ulb.items():
        a_ulb[key] = value.x

    a_uu_p = {}
    for key, value in var.a_uu_p.items():
        a_uu_p[key] = value.x

    a_pw_p = {}
    for key, value in var.a_pw_p.items():
        a_pw_p[key] = int(value.x)

    a_ps_p = {}
    for key, value in var.a_ps_p.items():
        a_ps_p[key] = int(value.x)

    # Returns
    st = state(s_ul, s_pw, s_ps)
    act = action(a_sc, a_rsc, a_uv, a_uvb, a_ul_p, a_ulb, a_uu_p, a_pw_p, a_ps_p)

    return (st, act)