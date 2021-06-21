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
    var_ue = {}
    var_uu = {}
    var_pw = {}
    var_ps = {}

    var_sc = {}
    var_rsc = {}
    var_uv = {}
    var_uu_p = {}
    var_pw_p = {}
    var_ps_p = {}

    # States / Upper Bounds
    # UE, UU
    for tp in itertools.product(indices['t'], indices['p']):
        ppe_upper_bounds = ppe_data[tp[1]].expected_units + ppe_data[tp[1]].deviation[1]
        var_ue[tp] = sub_model.addVar(name=f's_ue_{tp}', ub=4*ppe_upper_bounds, vtype=GRB.CONTINUOUS)
        var_uu[tp] = sub_model.addVar(name=f's_uu_{tp}', ub=4*ppe_upper_bounds, vtype=GRB.CONTINUOUS)
    # PW
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
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
        var_uv[tp] = sub_model.addVar(name=f'a_uv_{tp}', vtype=GRB.CONTINUOUS)
    # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        var_uu_p[tp] = sub_model.addVar(name=f'a_uu_p_{tp}', vtype=GRB.CONTINUOUS)
    # PW Hat
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        var_pw_p[mdc] = sub_model.addVar(name=f'a_pw_p_{mdc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps_p[tmdc] = sub_model.addVar(name=f'a_ps_p_{tmdc}', vtype=GRB.INTEGER)

    sub_vars = variables(
        var_ue, var_uu, var_pw, var_ps,
        var_sc, var_rsc, var_uv, var_uu_p, var_pw_p, var_ps_p
    )

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
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

    # Constraints
    # 1) Consistent Initial Resource Usage
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu[tp])
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdc[1], mdc[2])], var_ps[(tp[0], mdc[0], mdc[1], mdc[2])])
        sub_model.addConstr(expr == 0, name=f'consistent_usage_init_{tp}')

    # 2) Resource Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        sub_model.addConstr(var_uu_p[tp] <= var_ue[tp] + var_uv[tp], name=f'resource_constraint_{tp}')

    # 3) Bounds on Reschedules
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        # if ttpmdc[0] == ttpmdc[1]:
        sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        # elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            # sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        # elif ttpmdc[0] == 1 and ttpmdc[1] >= 3:
        #     sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 4) Number of people schedules/reschedules must be consistent
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
            expr.addTerms(model_param.cw**mdc[0], var.a_pw_p[mdc])                    
        
        # Cost of Waiting - Last Period
        for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
            expr.addTerms(model_param.cw**indices['m'][-1], var.a_ps_p[(tdc[0],indices['m'][-1],tdc[1],tdc[2])])     

        return(expr)
    def pref_earlier_appointment(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(model_param.cs**tmdc[0], var.a_sc[tmdc])
        return(expr)
    def reschedule_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[1] > ttpmdc[0]:
                expr.addTerms(1.5*model_param.cc, var.a_rsc[ttpmdc])
            elif ttpmdc[1] < ttpmdc[0]:
                expr.addTerms(-(0.5*model_param.cc), var.a_rsc[ttpmdc])

        return(expr)
    def goal_violation_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var.a_uv[tp])

        return(expr)
    
    # E[V] Function
    def b0_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        expr.addConstant((1-gamma) * betas['b0']['b_0'])
        return(expr)
    def b_ue_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        for tp in itertools.product(indices['t'], indices['p']):
            # When t is 1
            if tp[0] == 1:
                expr.addTerms(betas['ue'][tp], var.s_ue[tp])
                expr.addConstant(-gamma * betas['ue'][tp] * ppe_data[tp[1]].expected_units)
                expr.addTerms(-gamma * betas['ue'][tp], var.s_ue[tp])
                expr.addTerms(gamma * betas['ue'][tp], var.a_uu_p[tp])
                expr.addTerms(-gamma * betas['ue'][tp], var.a_uv[tp])

            # All other
            else:
                expr.addTerms(betas['ue'][tp], var.s_ue[tp])
                expr.addConstant(-gamma * betas['ue'][tp] * ppe_data[tp[1]].expected_units)
                    
        return(expr)
    def b_uu_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            # When t is T
            if tp[0] == indices['t'][-1]:
                expr.addTerms( betas['uu'][tp], var.s_uu[tp] )
            
            # All others
            else:
                expr.addTerms( betas['uu'][tp], var.s_uu[tp] )
                expr.addTerms( -betas['uu'][tp] * gamma, var.a_uu_p[(tp[0]+1, tp[1])] )
                
                # Change due to transition in complexity
                for mc in itertools.product(indices['m'], indices['c']):
                    for d in range(len(indices['d'])):

                        # When d is D
                        if d == len(indices['d'])-1: 
                            pass

                        # Otherwise
                        else:
                            transition_prob = transition[(mc[0], indices['d'][d], mc[1])]
                            usage_change = usage[(tp[1], indices['d'][d+1], mc[1])] - usage[(tp[1], indices['d'][d], mc[1])]
                            coeff = betas['uu'][tp] * gamma * transition_prob * usage_change
                            expr.addTerms( -coeff, var.a_ps_p[ (tp[0]+1, mc[0], indices['d'][d], mc[1]) ] )

        return(expr)
    def b_pw_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])

                # When m is 0
                if mc[0] == 0: 
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addConstant(-betas['pw'][mdc] * gamma * arrival[mdc[1], mdc[2]])

                # When m is M
                elif mc[0] == indices['m'][-1]:
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mm, mdc[1], mdc[2])])
           
                        # Transitioned In
                        if d != 0:
                            expr.addTerms(
                                -betas['pw'][mdc] * gamma * transition[( mm, indices['d'][d-1], mdc[2] )],
                                var.a_pw_p[( mm, indices['d'][d-1], mdc[2] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['pw'][mdc] * gamma * transition[( mm, mdc[1], mdc[2] )],
                                var.a_pw_p[( mm, mdc[1], mdc[2] )]
                            )

                # All others
                else:                   
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma, var.a_pw_p[(mdc[0]-1, mdc[1], mdc[2])])
           
                    # Transitioned In
                    if d != 0:
                        expr.addTerms(
                            -betas['pw'][mdc] * gamma * transition[( mdc[0]-1, indices['d'][d-1], mdc[2] )],
                            var.a_pw_p[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['pw'][mdc] * gamma * transition[( mdc[0]-1, mdc[1], mdc[2] )],
                            var.a_pw_p[( mdc[0]-1, mdc[1], mdc[2] )]
                        )

        return(expr)
    def b_ps_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])

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
                        if d != 0:
                            expr.addTerms(
                                -betas['ps'][tmdc] * gamma * transition[( mm, indices['d'][d-1], tmdc[3] )],
                                var.a_ps_p[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                betas['ps'][tmdc] * gamma * transition[( mm, tmdc[2], tmdc[3] )],
                                var.a_ps_p[( tmdc[0]+ 1, mm, tmdc[2], tmdc[3] )]
                            )
                
                # Everything Else
                else:
                    expr.addTerms(betas['ps'][tmdc], var.s_ps[tmdc])
                    expr.addTerms(-betas['ps'][tmdc] * gamma, var.a_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                    # Transitioned In
                    if d != 0:
                        expr.addTerms(
                            -betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )],
                            var.a_ps_p[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, tmdc[2], tmdc[3] )],
                            var.a_ps_p[( tmdc[0]+ 1, tmdc[1]-1, tmdc[2], tmdc[3] )]
                        )

        return(expr)
    
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(sub_vars, betas)
    pref_early = pref_earlier_appointment(sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(sub_vars, betas)
    b_ue_expr = b_ue_cost(sub_vars, betas)
    b_uu_expr = b_uu_costs(sub_vars, betas)
    b_pw_expr = b_pw_costs(sub_vars, betas)
    b_ps_expr = b_ps_costs(sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ue_expr + b_uu_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(-value_expr, GRB.MINIMIZE)
    else:
        sub_model.setObjective(cost_expr - value_expr, GRB.MINIMIZE)

    return sub_model, sub_vars

# Reads variables results and returns them
def generate_state_action(var: variables) -> Tuple[state, action]:
    
    # Extracts
    s_ue = {}
    for key, value in var.s_ue.items():
        s_ue[key] = value.x

    s_uu = {}
    for key, value in var.s_uu.items():
        s_uu[key] = value.x
        
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
    st = state(s_ue, s_uu, s_pw, s_ps)
    act = action(a_sc, a_rsc, a_uv, a_uu_p, a_pw_p, a_ps_p)

    return (st, act)