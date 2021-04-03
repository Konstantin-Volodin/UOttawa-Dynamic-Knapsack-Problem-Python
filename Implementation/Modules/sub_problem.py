from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Data Classes
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
@dataclass(frozen=True)
class variables:
    s_ue: Dict[ Tuple[str],gp.Var]
    s_uu: Dict[ Tuple[str],gp.Var]
    s_uv: Dict[ Tuple[str],gp.Var]
    s_pe: Dict[ Tuple[str],gp.Var]
    s_pw: Dict[ Tuple[str],gp.Var]
    s_ps: Dict[ Tuple[str],gp.Var]
    a_sc: Dict[ Tuple[str],gp.Var]
    a_rsc: Dict[ Tuple[str],gp.Var]

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
    var_uv = {}
    var_pw = {}
    var_pe = {}
    var_ps = {}
    var_sc = {}
    var_rsc = {}

    # States / Upper Bounds
    for tp in itertools.product(indices['t'], indices['p']):
        ppe_upper_bounds = ppe_data[tp[1]].expected_units + ppe_data[tp[1]].deviation[1]
        var_ue[tp] = sub_model.addVar(name=f's_ue_{tp}', ub=ppe_upper_bounds, vtype=GRB.INTEGER)
        var_uu[tp] = sub_model.addVar(name=f's_uu_{tp}', ub=ppe_upper_bounds*2, vtype=GRB.INTEGER)
        var_uv[tp] = sub_model.addVar(name=f's_uv_{tp}', ub=ppe_upper_bounds, vtype=GRB.INTEGER)
    for dc in itertools.product(indices['d'], indices['c']):
        var_pe[dc] = sub_model.addVar(name=f's_pe_{dc}', ub=4*arrival[dc], vtype=GRB.INTEGER)
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == 0: continue
        var_pw[mdc] = sub_model.addVar(name=f's_pw_{mdc}', ub=4*arrival[(mdc[1],mdc[2])], vtype=GRB.INTEGER)
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps[tmdc] = sub_model.addVar(name=f's_ps_{tmdc}', ub=4*arrival[(tmdc[2],tmdc[3])], vtype=GRB.INTEGER)
    # Actions
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = sub_model.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[2] == 0: continue
        var_rsc[ttpmdc] = sub_model.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)

    sub_vars = variables( var_ue, var_uu, var_uv, var_pe, var_pw, var_ps, var_sc, var_rsc)

    # Constraints
    # 1) PPE Capacity
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1],mdc[1],mdc[2])], var_sc[(tp[0], mdc[0],mdc[1],mdc[2])])
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            if tmdc[1] == 0: continue
            expr.addTerms(-usage[(tp[1],tmdc[2],tmdc[3])], var_rsc[(tmdc[0], tp[0], tmdc[1],tmdc[2],tmdc[3])])
        expr.addTerms(1, var_ue[tp])
        expr.addTerms(-1, var_uu[tp])
        expr.addTerms(1, var_uv[tp])
        sub_model.addConstr(expr >= 0, name=f'ppe_capacity_{tp}')

    # 2) Bounds on Reschedules
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[2] == 0: continue
        if ttpmdc[0] == ttpmdc[1]:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] == 1 and ttpmdc[1] >= 3:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 3) Cap on max schedule/reschedule wait time

    # 4) Number of people schedules/reschedules must be consistent
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        if tmdc[1] == 0: continue
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdc[0], tp[0], tmdc[1], tmdc[2], tmdc[3])])
        expr.addTerms(1, var_ps[tmdc])
        sub_model.addConstr(expr >= 0, f'consistent_resc_{(tmdc)}')
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdc[0], mdc[1], mdc[2])])
        if mdc[0] == 0:
            expr.addTerms(1, var_pe[(mdc[1], mdc[2])])
        if mdc[0] >= 1:
            expr.addTerms(1, var_pw[mdc])
        sub_model.addConstr(expr >= 0, f'consistent_sch_{(mdc)}')

    # Objective Function
    # Cost Function
    def wait_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
    
        # PW Cost
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']): 
            if mdc[0] == 0:
                expr.addTerms(model_param.cw**mdc[0], var.s_pe[(mdc[1], mdc[2])])
            elif mdc[0] >= 1:
                expr.addTerms(model_param.cw**mdc[0], var.s_pw[mdc])
        # PS Cost
        for tdc in itertools.product(indices['t'], indices['d'], indices['c']): 
            expr.addTerms(model_param.cw**indices['m'][-1], var.s_ps[(tdc[0], indices['m'][-1], tdc[1], tdc[2])])

        return(expr)
    def reschedule_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[2] == 0: continue
            if ttpmdc[1] > ttpmdc[0]:
                expr.addTerms(model_param.cc, var.a_rsc[ttpmdc])
            elif ttpmdc[1] < ttpmdc[0]:
                expr.addTerms(-model_param.cc, var.a_rsc[ttpmdc])

        return(expr)
    def goal_violation_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var.s_uv[tp])

        return(expr)
    
    # E[V] Function
    def b0_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        expr.addConstant(betas['b0']['b_0'])
        return(expr)
    def b_ue_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()
        
        for tp in itertools.product(indices['t'], indices['p']):
        
            # When t is 1
            if tp[0] == 1:
                
                # Default
                expr.addTerms(betas['ue'][tp], var.s_ue[tp])
                expr.addConstant(-betas['ue'][tp] * gamma * ppe_data[tp[1]].expected_units)
                
                # Previous Leftover
                expr.addTerms(-betas['ue'][tp] * gamma, var.s_ue[tp])
                expr.addTerms(betas['ue'][tp] * gamma, var.s_uu[tp])
                
                # New Usage
                for dc in itertools.product(indices['d'], indices['c']):
                    for m in itertools.product(indices['m']):
                        expr.addTerms( usage[(tp[1], dc[0], dc[1])],  var.a_sc[(tp[0],m[0], dc[0], dc[1])])
                    for tpm in itertools.product(indices['t'], indices['m']):
                        if tpm[1] == 0: continue
                        expr.addTerms( usage[(tp[1], dc[0], dc[1])], var.a_rsc[(tp[0],tpm[0], tpm[1], dc[0], dc[1])] )
                    for tm in itertools.product(indices['t'], indices['m']):
                        if tm[1] == 0: continue
                        expr.addTerms( -usage[(tp[1], dc[0], dc[1])], var.a_rsc[(tm[0],tp[0], tm[1], dc[0], dc[1])] )
            
            # When t > 1
            elif tp[0] >= 2:
                expr.addTerms(betas['ue'][tp], var.s_ue[tp])
                expr.addConstant(-betas['ue'][tp] * gamma * ppe_data[tp[1]].expected_units)
        
        return(expr)
    def b_uu_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):

            # when T
            if tp[0] == indices['t'][-1]:
                expr.addTerms( betas['uu'][tp], var.s_uu[tp] )
            
            # All others
            else:
                expr.addTerms( betas['uu'][tp], var.s_uu[tp] )
                expr.addTerms( -betas['uu'][tp] * gamma, var.s_uu[(tp[0]+1, tp[1])] )
                
                # Change due to transition in complexity
                for mc in itertools.product(indices['m'], indices['c']):
                    for d in range(len(indices['d'])):
                        # if mc[0] == 0: continue
                        if d == (len(indices['d']) - 1): continue
                        change_in_usage = usage[( tp[1], indices['d'][d+1], mc[1] )] - usage[( tp[1], indices['d'][d], mc[1] )]
                        expr.addTerms( 
                            -betas['uu'][tp] * gamma * transition[mdc] * (change_in_usage), 
                            var.s_ps[(tp[0], mc[0], indices['d'][d], mc[1])]
                        )
                
                # Change due to scheduling
                for mdc in itertools.product(indices['m'],indices['d'],indices['c']):
                    expr.addTerms( 
                        -betas['uu'][tp] * gamma * usage[(tp[1], mdc[1], mdc[2])] , 
                        var.a_sc[(tp[0]+1, mdc[0], mdc[1], mdc[2])] 
                    )
                for tmdc in itertools.product(indices['t'], indices['m'],indices['d'],indices['c']):
                    if not (tmdc[0]+1 in indices['t']): continue
                    if tmdc[1] == 0: continue
                    expr.addTerms( 
                        betas['uu'][tp] * gamma * usage[(tp[1], mdc[1], mdc[2])] , 
                        var.a_rsc[(tmdc[0]+1, tp[0], tmdc[1], tmdc[2], tmdc[3])] 
                    )
                for tpmdc in itertools.product(indices['t'], indices['m'],indices['d'],indices['c']):
                    if not (tpmdc[0]+1 in indices['t']): continue
                    if tpmdc[1] == 0: continue
                    expr.addTerms( 
                        -betas['uu'][tp] * gamma * usage[(tp[1], mdc[1], mdc[2])] , 
                        var.a_rsc[(tp[0], tpmdc[0]+1, tpmdc[1], tpmdc[2], tpmdc[3])] 
                    )

        return(expr)
    def b_uv_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(betas['uv'][tp], var.s_uv[tp])

        return (expr)
    def b_pe_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for dc in itertools.product(indices['d'], indices['c']):
            expr.addTerms(betas['pe'][dc], var.s_pe[dc])
            expr.addConstant(-betas['pe'][dc] * gamma * arrival[dc])

        return(expr)
    def b_pw_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])

                if mc[0] == 0: continue
                
                # When m = 1
                elif mc[0] == 1:
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addTerms(-betas['pw'][mdc] * gamma , var.s_pe[(mdc[1], mdc[2])])
                    for t in range(len(indices['t'])):
                        expr.addTerms(betas['pw'][mdc] * gamma, var.a_sc[(indices['t'][t], 0, mdc[1], mdc[2])])

                # When m = M
                elif mc[0] == indices['m'][-1]:
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    # Transition out
                    expr.addTerms(-betas['pw'][mdc] * gamma * (1-transition[(mdc[0]-1, mdc[1], mdc[2])]), var.s_pw[(mdc[0]-1, mdc[1], mdc[2])] )
                    expr.addTerms(-betas['pw'][mdc] * gamma * (1-transition[mdc]), var.s_pw[mdc] )
                    # Transitioned in
                    if d != 0:
                        expr.addTerms(-betas['pw'][mdc] * gamma * transition[(mdc[0]-1, indices['d'][d-1], mdc[2])], var.s_pw[(mdc[0]-1, indices['d'][d-1], mdc[2])] )
                        expr.addTerms(-betas['pw'][mdc] * gamma * transition[(mdc[0], indices['d'][d-1], mdc[2])], var.s_pw[(mdc[0], indices['d'][d-1], mdc[2])] )
                    # Scheduled
                    for t in indices['t']:
                        expr.addTerms(betas['pw'][mdc] * gamma, var.a_sc[t, mdc[0]-1, mdc[1], mdc[2]])
                        expr.addTerms(betas['pw'][mdc] * gamma, var.a_sc[t, mdc[0], mdc[1], mdc[2]])

                # All others
                else:                   
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    # Transition out
                    expr.addTerms(-betas['pw'][mdc] * gamma * (1-transition[(mdc[0]-1, mdc[1], mdc[2])]), var.s_pw[(mdc[0]-1, mdc[1], mdc[2])] )
                    # Transitioned in
                    if d != 0:
                        expr.addTerms(-betas['pw'][mdc] * gamma * transition[(mdc[0]-1, indices['d'][d-1], mdc[2])], var.s_pw[(mdc[0]-1, indices['d'][d-1], mdc[2])] )
                    # Scheduled
                    for t in indices['t']:
                        expr.addTerms(betas['pw'][mdc] * gamma, var.a_sc[t, mdc[0]-1, mdc[1], mdc[2]])
                        
        return(expr)
    def b_ps_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])
                if tmc[1] == 0: continue

                # When t is T
                if tmdc[0] == indices['t'][-1]:
                    expr.addTerms( betas['ps'][tmdc], var.s_ps[tmdc] )

                # All other times
                else:
                    # When m is M
                    if tmdc[1] == indices['m'][-1]:
                        # Baseline
                        expr.addTerms( betas['ps'][tmdc], var.s_ps[tmdc] )
                        # Transition in difficulties
                        for mm in indices['m'][-2:]:
                            expr.addTerms( -betas['ps'][tmdc]*gamma * (1 - transition[( mm, tmdc[2], tmdc[3] )]), var.s_ps[ ( tmdc[0]+1, mm, tmdc[2], tmdc[3] ) ] )
                            if d != 0: 
                                expr.addTerms( -betas['ps'][tmdc] * gamma * transition[( mm, indices['d'][d-1], tmdc[3] )], var.s_ps[ (tmdc[0]+1, mm, indices['d'][d-1], tmdc[3]) ] )
                            
                            # Scheduling / Rescheduling
                            expr.addTerms( betas['ps'][tmdc] * gamma, var.a_sc[ (tmdc[0]+1, mm, tmdc[2], tmdc[3]) ] )
                            for t in indices['t']:
                                expr.addTerms( betas['ps'][tmdc] * gamma, var.a_rsc[ (tmdc[0]+1, t, mm, tmdc[2], tmdc[3]) ] )
                                expr.addTerms( -betas['ps'][tmdc] * gamma, var.a_rsc[ (t, tmdc[0]+1, mm, tmdc[2], tmdc[3]) ] )
                                
                    # When m is 0
                    elif tmdc[1] == 0: continue

                    # All other Ms
                    else:
                        # Baseline
                        expr.addTerms( betas['ps'][tmdc], var.s_ps[tmdc] )
                        # Transition in difficulties
                        if tmdc[1] >= 2:
                            expr.addTerms( -betas['ps'][tmdc]*gamma * (1 - transition[( tmdc[1]-1, tmdc[2], tmdc[3] )]), var.s_ps[ ( tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3] ) ] )
                            if d != 0: 
                                expr.addTerms( -betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )], var.s_ps[ (tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3]) ] )
                        
                        # Scheduling / Rescheduling
                        expr.addTerms( betas['ps'][tmdc] * gamma, var.a_sc[ (tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3]) ] )
                        if tmdc[1] >= 2:
                            for t in indices['t']:
                                expr.addTerms( betas['ps'][tmdc] * gamma, var.a_rsc[ (tmdc[0]+1, t, tmdc[1]-1, tmdc[2], tmdc[3]) ] )
                                expr.addTerms( -betas['ps'][tmdc] * gamma, var.a_rsc[ (t, tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3]) ] )


        return(expr)
    
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost(sub_vars, betas)
    rescheduling_cost_expr = reschedule_cost(sub_vars, betas)
    goal_vio_cost_expr = goal_violation_cost(sub_vars, betas)
    cost_expr = gp.LinExpr(wait_cost_expr + rescheduling_cost_expr + goal_vio_cost_expr)
    
        # Value
    b0_expr = b0_cost(sub_vars, betas)
    b_ue_expr = b_ue_cost(sub_vars, betas)
    b_uu_expr = b_uu_costs(sub_vars, betas)
    b_uv_expr = b_uv_costs(sub_vars, betas)
    b_pe_expr = b_pe_costs(sub_vars, betas)
    b_pw_expr = b_pw_costs(sub_vars, betas)
    b_ps_expr = b_ps_costs(sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ue_expr + b_uu_expr + b_uv_expr + b_pe_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(value_expr, GRB.MAXIMIZE)
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
    
    s_uv = {}
    for key, value in var.s_uv.items():
        s_uv[key] = value.x
    
    s_pw = {}
    for key, value in var.s_pw.items():
        s_pw[key] = value.x
    
    s_pe = {}
    for key, value in var.s_pe.items():
        s_pe[key] = value.x
    
    s_ps = {}
    for key, value in var.s_ps.items():
        s_ps[key] = value.x
    
    a_sc = {}
    for key, value in var.a_sc.items():
        a_sc[key] = value.x

    a_rsc = {}
    for key, value in var.a_rsc.items():
        a_rsc[key] = value.x

    # Returns
    st = state(s_ue, s_uu, s_uv, s_pe, s_pw, s_ps)
    act = action(a_sc, a_rsc)

    return (st, act)