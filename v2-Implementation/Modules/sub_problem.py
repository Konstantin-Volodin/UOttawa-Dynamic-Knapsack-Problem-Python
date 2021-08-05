from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from Modules.data_classes import state, action, variables

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Generate sub problem model
def generate_sub_model(input_data, betas, phase1 = False):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage
    M = model_param.M
    gamma = model_param.gamma

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
        var_ul[p] = sub_model.addVar(name=f's_ul_{p}', ub=5*ppe_data[p[0]].expected_units, vtype=GRB.CONTINUOUS)
    # PW
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == indices['m'][-1]:
            var_pw[mdc] = sub_model.addVar(name=f's_pw_{mdc}', ub=20*arrival[(mdc[1],mdc[2])], vtype=GRB.INTEGER)
        else:
            var_pw[mdc] = sub_model.addVar(name=f's_pw_{mdc}', ub=4*arrival[(mdc[1],mdc[2])], vtype=GRB.INTEGER)
    # PS
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps[tmdc] = sub_model.addVar(name=f's_ps_{tmdc}', ub=4*arrival[(tmdc[2],tmdc[3])], vtype=GRB.INTEGER)

    # Actions
    # SC
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = sub_model.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = sub_model.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)
    # UV
    for tp in itertools.product(indices['t'], indices['p']):
        var_uv[tp] = sub_model.addVar(name=f'a_uv_{tp}', ub=10*ppe_data[p[0]].expected_units, vtype=GRB.CONTINUOUS)

    
    # UL Hat & UL B
    for p in itertools.product(indices['p']):
        var_ul_p[p] = sub_model.addVar(name=f'a_ul_p_{p}', vtype=GRB.CONTINUOUS)
        var_ulb[p] = sub_model.addVar(name=f'a_ulb_{p}', vtype=GRB.BINARY)
    # UU Hat & UV B
    for tp in itertools.product(indices['t'], indices['p']):
        var_uu_p[tp] = sub_model.addVar(name=f'a_uu_p_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = sub_model.addVar(name=f'a_uvb_{tp}', vtype=GRB.BINARY)
    
    # PW Hat
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        var_pw_p[mdc] = sub_model.addVar(name=f'a_pw_p_{mdc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps_p[tmdc] = sub_model.addVar(name=f'a_ps_p_{tmdc}', vtype=GRB.INTEGER)

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
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdc[1], mdc[2])], var_ps_p[(tp[0], mdc[0], mdc[1], mdc[2])])
        sub_model.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdc])
        expr.addTerms(-1, var_pw[mdc])
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdc[0], mdc[1], mdc[2])])
        sub_model.addConstr(expr == 0, name=f'pw_hat_{mdc}')
        # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdc])
        expr.addTerms(-1, var_ps[tmdc])
        expr.addTerms(-1, var_sc[tmdc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdc[0], tmdc[1], tmdc[2], tmdc[3])])
            expr.addTerms(1, var_rsc[(tmdc[0], tp, tmdc[1], tmdc[2], tmdc[3])])
        sub_model.addConstr(expr == 0, name=f'ps_hat_{tmdc}')
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
        sub_model.addConstr(var_ul_p[p] >= 0)
        sub_model.addConstr(var_ul_p[p] >= input_data.ppe_data[p[0]].expected_units + var_ul[p] - var_uu_p[(1, p[0])])
        sub_model.addConstr(var_ul_p[p] <= M * var_ulb[p])
        sub_model.addConstr(var_ul_p[p] <= input_data.ppe_data[p[0]].expected_units + var_ul[p] - var_uu_p[(1, p[0])] + (M * (1-var_ulb[p])))

    # Constraints
    # 1) Resource Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            # sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')
            sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_ul[(tp[1],)] + var_uv[tp], name=f'resource_constraint_{tp}')
        else:
            sub_model.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')

    # 2) Bounds on Reschedules
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] == ttpmdc[1] == 1:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdc[0], tp[0], tmdc[1], tmdc[2], tmdc[3])])
        expr.addTerms(1, var_ps[tmdc])
        sub_model.addConstr(expr >= 0, f'consistent_resc_{(tmdc)}')
        # Scheduled
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdc[0], mdc[1], mdc[2])])
        expr.addTerms(1, var_pw[mdc])
        sub_model.addConstr(expr >= 0, f'consistent_sch_{(mdc)}')

    # Objective Function
    # Cost Function
    def wait_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        # Cost of Waiting
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):  
            expr.addTerms(model_param.cw, var.a_pw_p[mdc])                    
        
        # Cost of Waiting - Last Period
        # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        #     expr.addTerms(model_param.cw**(indices['m'][-1]+1), var.a_ps_p[(tdc[0],indices['m'][-1],tdc[1],tdc[2])])     

        return expr
    def pref_earlier_appointment(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(model_param.cs[tmdc[0]], var.a_sc[tmdc])

        return expr
    def reschedule_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[0] > ttpmdc[1]: # Good Reschedule
                difference = ttpmdc[0] - ttpmdc[1]
                expr.addTerms(-(model_param.cs[difference] - model_param.cc), var.a_rsc[ttpmdc])
            elif ttpmdc[0] < ttpmdc[1]: # Bad Reschedule
                difference = ttpmdc[1] - ttpmdc[0]
                expr.addTerms(model_param.cs[difference] + model_param.cc, var.a_rsc[ttpmdc])

        return expr
    def goal_violation_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var.a_uv[tp])

        return(expr)
    
    # E[V] Function
    def b0_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        expr.addConstant((1-gamma) * betas['b0']['b_0'])
        return expr
    def b_ul_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        for p in itertools.product(indices['p']):    
            if ppe_data[p[0]].ppe_type == 'carry-over':
                expr.addTerms(betas['ul'][p], var.s_ul[p])
                expr.addTerms(- (betas['ul'][p] * gamma), var.a_ul_p[p])    
            elif ppe_data[p[0]].ppe_type == 'non-carry-over':
                expr.addTerms(betas['ul'][p], var.s_ul[p])
                    
        return expr
    def b_pw_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])
                dc = (indices['d'][d], mc[1])

                # When m is 0
                if mc[0] == 0: 
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addConstant(-betas['pw'][mdc] * gamma * arrival[(mdc[1], mdc[2])])

                # When m is M
                elif mc[0] == indices['m'][-1]:
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mm, mdc[1], mdc[2])])
           
                        # Transitioned In
                        if d != 0 & (mm >= transition[dc].wait_limit+1):
                            expr.addTerms(
                                -betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                                var.a_pw_p[( mm, indices['d'][d-1], mdc[2] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                                var.a_pw_p[( mm, mdc[1], mdc[2] )]
                            )

                # When m is less than TL_dc
                elif mc[0] <= (transition[dc].wait_limit - 1):
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mdc[0]-1, mdc[1], mdc[2])])

                # All others
                else:                   
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mdc[0]-1, mdc[1], mdc[2])])
           
                    # Transitioned In
                    if d != 0 & (mdc[0] >= transition[dc].wait_limit+1):
                        expr.addTerms(
                            -betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                            var.a_pw_p[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                            var.a_pw_p[( mdc[0]-1, mdc[1], mdc[2] )]
                        )

        return expr
    def b_ps_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])
                dc = (indices['d'][d], tmc[2])

                # When m is 0
                if tmdc[1] == 0: 
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                # When t is T
                elif tmdc[0] == indices['t'][-1]:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                # when m is M
                elif tmdc[1] == indices['m'][-1]:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, mm, tmdc[2], tmdc[3])])
           
                        # Transitioned In
                        if d != 0 & (mm >= transition[dc].wait_limit+1):
                            expr.addTerms(
                                -betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                                var.a_ps_p[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                                var.a_ps_p[( tmdc[0]+ 1, mm, tmdc[2], tmdc[3] )]
                            )
                
                # When m is less than TL_dc
                elif tmdc[1] <= (transition[dc].wait_limit - 1):
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])
                    expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                # Everything Else
                else:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])
                    expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                    # Transitioned In
                    if d != 0 & (tmdc[1] >= transition[dc].wait_limit+1):
                        expr.addTerms(
                            -betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                            var.a_ps_p[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                            var.a_ps_p[( tmdc[0]+ 1, tmdc[1]-1, tmdc[2], tmdc[3] )]
                        )

        return expr
    
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(sub_vars, betas)
    pref_early = pref_earlier_appointment(sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(sub_vars, betas)
    b_ul_expr = b_ul_cost(sub_vars, betas)
    b_pw_expr = b_pw_costs(sub_vars, betas)
    b_ps_expr = b_ps_costs(sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ul_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(-value_expr, GRB.MINIMIZE)
    else:
        sub_model.setObjective(cost_expr - value_expr, GRB.MINIMIZE)

    return sub_model, sub_vars

# Updates sub problem model
def update_sub_model(input_data, model, variables, betas, phase1 = False):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage
    M = model_param.M
    gamma = model_param.gamma

    sub_model = model
    sub_vars = variables

    # Objective Function
    # Cost Function
    def wait_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        # Cost of Waiting
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):  
            expr.addTerms(model_param.cw, var.a_pw_p[mdc])                    
        
        # Cost of Waiting - Last Period
        # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        #     expr.addTerms(model_param.cw**(indices['m'][-1]+1), var.a_ps_p[(tdc[0],indices['m'][-1],tdc[1],tdc[2])])     

        return expr
    def pref_earlier_appointment(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(model_param.cs[tmdc[0]], var.a_sc[tmdc])

        return expr
    def reschedule_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[0] > ttpmdc[1]: # Good Reschedule
                difference = ttpmdc[0] - ttpmdc[1]
                expr.addTerms(-(model_param.cs[difference] - model_param.cc), var.a_rsc[ttpmdc])
            elif ttpmdc[0] < ttpmdc[1]: # Bad Reschedule
                difference = ttpmdc[1] - ttpmdc[0]
                expr.addTerms(model_param.cs[difference] + model_param.cc, var.a_rsc[ttpmdc])

        return expr
    def goal_violation_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var.a_uv[tp])

        return(expr)
    
    # E[V] Function
    def b0_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        expr.addConstant((1-gamma) * betas['b0']['b_0'])
        return expr
    def b_ul_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        for p in itertools.product(indices['p']):    
            if ppe_data[p[0]].ppe_type == 'carry-over':
                expr.addTerms(betas['ul'][p], var.s_ul[p])
                expr.addTerms(- (betas['ul'][p] * gamma), var.a_ul_p[p])    
            elif ppe_data[p[0]].ppe_type == 'non-carry-over':
                expr.addTerms(betas['ul'][p], var.s_ul[p])
                    
        return expr
    def b_pw_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])
                dc = (indices['d'][d], mc[1])

                # When m is 0
                if mc[0] == 0: 
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addConstant(-betas['pw'][mdc] * gamma * arrival[(mdc[1], mdc[2])])

                # When m is M
                elif mc[0] == indices['m'][-1]:
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mm, mdc[1], mdc[2])])
           
                        # Transitioned In
                        if d != 0 & (mm >= transition[dc].wait_limit+1):
                            expr.addTerms(
                                -betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                                var.a_pw_p[( mm, indices['d'][d-1], mdc[2] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                                var.a_pw_p[( mm, mdc[1], mdc[2] )]
                            )

                # When m is less than TL_dc
                elif mc[0] <= (transition[dc].wait_limit - 1):
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mdc[0]-1, mdc[1], mdc[2])])

                # All others
                else:                   
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mdc[0]-1, mdc[1], mdc[2])])
           
                    # Transitioned In
                    if d != 0 & (mdc[0] >= transition[dc].wait_limit+1):
                        expr.addTerms(
                            -betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                            var.a_pw_p[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['pw'][mdc] * gamma * transition[dc].transition_rate,
                            var.a_pw_p[( mdc[0]-1, mdc[1], mdc[2] )]
                        )

        return expr
    def b_ps_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])
                dc = (indices['d'][d], tmc[2])

                # When m is 0
                if tmdc[1] == 0: 
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                # When t is T
                elif tmdc[0] == indices['t'][-1]:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                # when m is M
                elif tmdc[1] == indices['m'][-1]:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, mm, tmdc[2], tmdc[3])])
           
                        # Transitioned In
                        if d != 0 & (mm >= transition[dc].wait_limit+1):
                            expr.addTerms(
                                -betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                                var.a_ps_p[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                                var.a_ps_p[( tmdc[0]+ 1, mm, tmdc[2], tmdc[3] )]
                            )
                
                # When m is less than TL_dc
                elif tmdc[1] <= (transition[dc].wait_limit - 1):
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])
                    expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                # Everything Else
                else:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])
                    expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                    # Transitioned In
                    if d != 0 & (tmdc[1] >= transition[dc].wait_limit+1):
                        expr.addTerms(
                            -betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                            var.a_ps_p[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['ps'][tmdc] * gamma * transition[dc].transition_rate,
                            var.a_ps_p[( tmdc[0]+ 1, tmdc[1]-1, tmdc[2], tmdc[3] )]
                        )

        return expr
    
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(sub_vars, betas)
    pref_early = pref_earlier_appointment(sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(sub_vars, betas)
    b_ul_expr = b_ul_cost(sub_vars, betas)
    b_pw_expr = b_pw_costs(sub_vars, betas)
    b_ps_expr = b_ps_costs(sub_vars, betas)
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