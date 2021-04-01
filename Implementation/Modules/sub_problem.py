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
class constraint_parameter:
    lhs_param: Dict[ Tuple[str],float ]
    rhs_param: Dict[ Tuple[str],float ]
    sign: Dict[ Tuple[str],str ]
    name:  Dict[ Tuple[str],str ]

def generate_sub_model(input_data, betas):
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

    var_ue = {}
    var_uu = {}
    var_uv = {}
    var_pw = {}
    var_pe = {}
    var_ps = {}
    var_sc = {}
    var_rsc = {}
    variables = {
        
    }

    # Decision Variables
    # States / Upper Bounds
    for tp in itertools.product(indices['t'], indices['p']):
        ppe_upper_bounds = ppe_data[tp[1]].expected_units + ppe_data[tp[1]].deviation[1]
        var_ue[tp] = sub_model.addVar(name=f's_ue_{tp}', ub=ppe_upper_bounds)
        var_uu[tp] = sub_model.addVar(name=f's_uu_{tp}', ub=ppe_upper_bounds*2)
        var_uv[tp] = sub_model.addVar(name=f's_uv_{tp}', ub=ppe_upper_bounds)
    for dc in itertools.product(indices['d'], indices['c']):
        var_pe[dc] = sub_model.addVar(name=f's_pe_{dc}', ub=4*arrival[dc])
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == 0: continue
        var_pw[mdc] = sub_model.addVar(name=f's_pw_{mdc}', ub=4*arrival[(mdc[1],mdc[2])])
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps[tmdc] = sub_model.addVar(name=f's_ps_{tmdc}', ub=4*arrival[(tmdc[2],tmdc[3])])
    # Actions
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = sub_model.addVar(name=f'a_sc_{tmdc}')
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[2] == 0: continue
        var_rsc[ttpmdc] = sub_model.addVar(name=f'a_rsc_{ttpmdc}')

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

    # Main Objective Function
    # Cost Function
    cost_expr = gp.LinExpr()
    # PW Cost
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']): 
        if mdc[0] == 0:
            cost_expr.addTerms(model_param.cw**mdc[0], var_pe[(mdc[1], mdc[2])])
        elif mdc[0] >= 1:
            cost_expr.addTerms(model_param.cw**mdc[0], var_pw[mdc])
    # PS Cost
    for tdc in itertools.product(indices['t'], indices['d'], indices['c']): 
        cost_expr.addTerms(model_param.cw**indices['m'][-1], var_ps[(tdc[0], indices['m'][-1], tdc[1], tdc[2])])
    # RSC Cost
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[2] == 0: continue
        if ttpmdc[1] > ttpmdc[0]:
            cost_expr.addTerms(model_param.cc, var_rsc[ttpmdc])
        elif ttpmdc[1] < ttpmdc[0]:
            cost_expr.addTerms(-model_param.cc, var_rsc[ttpmdc])
    # Violation Cost
    for tp in itertools.product(indices['t'], indices['p']):
        cost_expr.addTerms(M, var_uv[tp])
    
    # E[V] Functions
    def b0_cost(input_data, variables)
    expected_value_expr = gp.LinExpr()
    # Beta 0
    expected_value_expr.addConstant(betas['b0']['b_0'])
    # Beta ue
    for tp in itertools.product(indices['t'], indices['p']):
        # when t is 1
        if tp[0] == 1:
            # Default
            expected_value_expr.addTerms(betas['ue'][tp], var_ue[tp])
            expected_value_expr.addConstant(-betas['ue'][tp] * gamma * ppe_data[tp[1]].expected_units)
            # Previous Leftover
            expected_value_expr.addTerms(-betas['ue'][tp] * gamma, var_ue[tp])
            expected_value_expr.addTerms(betas['ue'][tp] * gamma, var_uu[tp])
            # New Usage
            for dc in itertools.product(indices['d'], indices['c']):
                for m in itertools.product(indices['m']):
                    expected_value_expr.addTerms( usage[(tp[1], dc[0], dc[1])],  var_sc[(tp[0],m[0], dc[0], dc[1])])
                for tpm in itertools.product(indices['t'], indices['m']):
                    if tpm[1] == 0: continue
                    expected_value_expr.addTerms( usage[(tp[1], dc[0], dc[1])], var_rsc[(tp[0],tpm[0], tpm[1], dc[0], dc[1])] )
                for tm in itertools.product(indices['t'], indices['m']):
                    if tm[1] == 0: continue
                    expected_value_expr.addTerms( -usage[(tp[1], dc[0], dc[1])], var_rsc[(tm[0],tp[0], tm[1], dc[0], dc[1])] )
        # when t > 1
        elif tp[0] >= 2:
            expected_value_expr.addTerms(betas['ue'][tp], var_ue[tp])
            expected_value_expr.addConstant(-betas['ue'][tp] * gamma * ppe_data[tp[1]].expected_units)
    # Beta uu
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == indices['t'][-1]:
            expected_value_expr.addTerms( betas[tp], var_uu[tp] )
        else:
            expected_value_expr.addTerms( betas[tp], var_uu[tp] )
            expected_value_expr.addTerms( -betas[tp] * gamma, var_uu[(tp[0]+1, tp[1])] )
    # Beta uv
    # Beta pe
    # Beta pw
    # Beta ps

    sub_model.setObjective(cost_expr - expected_value_expr, GRB.MINIMIZE)

    return sub_model