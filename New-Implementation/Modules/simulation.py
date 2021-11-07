from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from Modules.data_classes import state, action, variables

from multiprocessing import Pool
from tqdm import tqdm, trange
from copy import deepcopy
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Initializes State ( Everything Empty )
def initial_state(input_data) -> state:
    # Input Data
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data

    # Patient States
    pw = {}
    for mdkc in itertools.product(indices['m'],indices['d'], indices['k'], indices['c']):
        if mdkc[0] == 0:
            pw[mdkc] = round(arrival[(mdkc[1], mdkc[2], mdkc[3])]) 
        else:
            pw[mdkc] = 0
    ps = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps[tmdkc] = 0

    # Unit States
    ul = {}
    for p in itertools.product(indices['p']):
        ul[p] = 0

    init_state = state(ul, pw, ps)

    return init_state
# Initializes Action ( Everything Empty )
def initial_action(input_data) -> action:
    indices = input_data.indices

    # Action
    sc = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        sc[tmdkc] = 0
    rsc = {}
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        rsc[ttpmdkc] = 0
    
    # Violation
    uv = {}
    uvb = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uv[tp] = 0
        uvb[tp] = 0

    # Units left over 
    ul_p = {}
    ulb = {}
    for p in itertools.product(indices['p']):
        ul_p[p] = 0
        ulb[p] = 0

    # Auxiliary variables
    uu_p = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uu_p[tp] = 0
    pw_p = {}
    for mdc in itertools.product(indices['m'],indices['d'], indices['k'], indices['c']):
        pw_p[mdc] = 0
    ps_p = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps_p[tmdkc] = 0

    # Returns
    init_action = action(sc, rsc, uv, uvb, ul_p, ulb,  uu_p, pw_p, ps_p)
    return(init_action)

# Executes Action
def execute_action(input_data, state, action) -> state:
    
    ppe_data = input_data.ppe_data
    indices = input_data.indices
    new_state = deepcopy(state)

    # Units Left Over
    for p in itertools.product(indices['p']):    
        if ppe_data[p[0]].ppe_type == 'carry-over':
            new_state.ul_p[p] = action.ul_p_p[p]    
        elif ppe_data[p[0]].ppe_type == 'non-carry-over':
            new_state.ul_p[p] = 0    

    # Patients Scheduled
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        new_state.ps_tmdkc[tmdkc]= action.ps_p_tmdkc[tmdkc]
                
    # Patients Waiting
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        new_state.pw_mdkc[mdkc]= action.pw_p_mdkc[mdkc]

    return(new_state)
# Generates cost of state-action
def state_action_cost(input_data, state, action) -> float:
    # Initializes
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
# Executes Transition to next state
def execute_transition(input_data, state, action) -> state:

    transition = input_data.transition
    indices = input_data.indices
    new_state = deepcopy(state)

    # UL
    for p in itertools.product(indices['p']):
        deviation = np.random.uniform(
            input_data.ppe_data[p[0]].deviation[0],
            input_data.ppe_data[p[0]].deviation[1]
        )

        if input_data.ppe_data[p[0]].ppe_type == 'non-carry-over':
            new_state.ul_p[p] = deviation
        else:
            new_state.ul_p[p] = deviation + action.ul_p_p[p]

    # PW
        # Generates New Arrivals, Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        if mdkc[0] == 0:
            new_state.pw_mdkc[mdkc] = np.random.poisson(input_data.arrival[(mdkc[1], mdkc[2], mdkc[3])])
        elif mdkc[0] == indices['m'][-1]:
            new_state.pw_mdkc[mdkc] += state.pw_mdkc[(mdkc[0] - 1, mdkc[1], mdkc[2], mdkc[3])]
        else:
            new_state.pw_mdkc[mdkc] = state.pw_mdkc[(mdkc[0] - 1, mdkc[1], mdkc[2], mdkc[3])]

        # Patient Complexity Transitions
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):
                mdkc = (mc[0], indices['d'][d], indices['k'][k], mc[1])

                if mc[0] <= (transition.wait_limit[mc[1]]-1): continue
                if indices['d'][d] == indices['d'][-1]: continue

                # Complexity Change
                tr_rate_d = transition.transition_rate_comp[(indices['d'][d], mc[1])]
                tr_out_d = np.random.binomial(new_state.pw_mdkc[mdkc], tr_rate_d)

                new_state.pw_mdkc[mdkc] -= tr_out_d
                new_state.pw_mdkc[(mdkc[0], indices['d'][d+1], mdkc[2], mdkc[3])] += tr_out_d

                # # Change in PPE Usage
                # for p in indices['p']:
                #     ppe_change = input_data.usage[(p, indices['d'][d+1], tmc[2])] - input_data.usage[(p, indices['d'][d], tmc[2])]
                #     new_state.uu_tp[(tmc[0], p)] += ppe_change*patients_transitioned
    
        # Patient Priority Transitions
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):
                mdkc = (mc[0], indices['d'][d], indices['k'][k], mc[1])

                if mc[0] <= (transition.wait_limit[mc[1]]-1): continue
                if indices['k'][k] == indices['k'][-1]: continue
                
                # Priority Change
                tr_rate_k = transition.transition_rate_pri[(indices['k'][k], mc[1])]
                tr_out_k = np.random.binomial(new_state.pw_mdkc[mdkc], tr_rate_k)

                new_state.pw_mdkc[mdkc] -= tr_out_k
                new_state.pw_mdkc[(mdkc[0], mdkc[1], indices['k'][k+1], mdkc[3])] += tr_out_k

    # PS
        # Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for tmdkc in itertools.product(indices['t'], reversed(indices['m']), indices['d'], indices['k'], indices['c']):
        if tmdkc[0] == indices['t'][-1]:
            new_state.ps_tmdkc[tmdkc] = 0
        elif tmdkc[1] == 0:
            new_state.ps_tmdkc[tmdkc] = 0
        elif tmdkc[1] == indices['m'][-1]:
            new_state.ps_tmdkc[tmdkc] = action.ps_p_tmdkc[(tmdkc[0] + 1, tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])] + action.ps_p_tmdkc[(tmdkc[0] + 1, tmdkc[1] - 1, tmdkc[2], tmdkc[3], tmdkc[4])]
        else:
            new_state.ps_tmdkc[tmdkc] = action.ps_p_tmdkc[(tmdkc[0] + 1, tmdkc[1] - 1, tmdkc[2], tmdkc[3], tmdkc[4])]

        # Patient Complexity Transitions
    for tmc in itertools.product(indices['t'], reversed(indices['m']), indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):
                tmdkc = (tmc[0], tmc[1], indices['d'][d], indices['k'][k], tmc[2])

                if tmdkc[1] <= (transition.wait_limit[tmdkc[4]]-1): continue
                if indices['d'][d] == indices['d'][-1]: continue

                # Complexity Change
                tr_rate_d = transition.transition_rate_comp[(indices['d'][d], tmdkc[4])]
                tr_out_d = np.random.binomial(new_state.ps_tmdkc[tmdkc], tr_rate_d)

                new_state.ps_tmdkc[tmdkc] -= tr_out_d
                new_state.ps_tmdkc[(tmdkc[0], tmdkc[1], indices['d'][d+1], tmdkc[3], tmdkc[4])] += tr_out_d

    
        # Patient Priority Transitions
    for tmc in itertools.product(indices['t'], reversed(indices['m']), indices['c']):
        for d in range(len(indices['d'])):
            for k in range(len(indices['k'])):
                tmdkc = (tmc[0], tmc[1], indices['d'][d], indices['k'][k], tmc[2])

                if tmdkc[1] <= (transition.wait_limit[tmdkc[4]]-1): continue
                if indices['k'][k] == indices['k'][-1]: continue

                # Complexity Change
                tr_rate_k = transition.transition_rate_pri[(indices['k'][k], tmdkc[4])]
                tr_out_k = np.random.binomial(new_state.ps_tmdkc[tmdkc], tr_rate_k)

                new_state.ps_tmdkc[tmdkc] -= tr_out_k
                new_state.ps_tmdkc[(tmdkc[0], tmdkc[1], tmdkc[2], indices['k'][k+1], tmdkc[4])] += tr_out_k

    return new_state

# Various Policies
def myopic_policy(input_data, state) -> action:
    
    # Input Data
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    usage = input_data.usage
    M = input_data.model_param.M

    # State Data
    ul = state.ul_p
    pw = state.pw_mdkc
    ps = state.ps_tmdkc

    # Initializes model
    myopic = gp.Model('Myopic Policy')
    myopic.Params.LogToConsole = 0

    # Decision Variables
    var_sc = {}
    var_rsc = {}
    var_uv = {}

    var_uvb = {}
    var_ul_p = {}
    var_ulb = {}
    var_uu_p = {}
    var_pw_p = {}
    var_ps_p = {}

    # SC
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_sc[tmdkc] = myopic.addVar(name=f'a_sc_{tmdkc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_rsc[ttpmdkc] = myopic.addVar(name=f'a_rsc_{ttpmdkc}', vtype=GRB.INTEGER)
    # UV
    for tp in itertools.product(indices['t'], indices['p']):
        var_uv[tp] = myopic.addVar(name=f'a_uv_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = myopic.addVar(name=f'a_uvb{tp}', vtype=GRB.BINARY)

    # UL Hat & UL B
    for p in itertools.product(indices['p']):
        var_ul_p[p] = myopic.addVar(name=f'a_ul_p_{p}', vtype=GRB.CONTINUOUS, obj=0)
        var_ulb[p] = myopic.addVar(name=f'a_ulb_{p}', vtype=GRB.BINARY, obj=0)
    # UU Hat & UV B
    for tp in itertools.product(indices['t'], indices['p']):
        var_uu_p[tp] = myopic.addVar(name=f'a_uu_p_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = myopic.addVar(name=f'a_uvb_{tp}', vtype=GRB.BINARY)
    
    # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        var_pw_p[mdkc] = myopic.addVar(name=f'a_pw_p_{mdkc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_ps_p[tmdkc] = myopic.addVar(name=f'a_ps_p_{tmdkc}', vtype=GRB.INTEGER)

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdkc[1], mdkc[3])], var_ps_p[(tp[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        myopic.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdkc])
        expr.addConstant(round(-state.pw_mdkc[mdkc],0))
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        myopic.addConstr(expr == 0, name=f'pw_hat_{mdkc}')
        # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdkc])
        expr.addConstant(-state.ps_tmdkc[tmdkc])
        expr.addTerms(-1, var_sc[tmdkc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdkc[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
            expr.addTerms(1, var_rsc[(tmdkc[0], tp, tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        myopic.addConstr(expr == 0, name=f'ps_hat_{tmdkc}')
        # UV Maximum function
    for tp in itertools.product(indices['t'], indices['p']):
        myopic.addConstr(var_uv[tp] <= M * var_uvb[tp], name=f'uv_max_1_{tp}')
        
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        expr.addConstant(-input_data.ppe_data[tp[1]].expected_units)
        expr.addConstant(M)
        expr.addTerms(-M, var_uvb[tp])
        if tp[0] == 1:
            expr.addConstant(-state.ul_p[(tp[1],)])
        myopic.addConstr(var_uv[tp] <= expr, name=f'uv_max_2_{tp}')
        # UL Maximum function
    for p in itertools.product(indices['p']):
        myopic.addConstr(var_ul_p[p] >= 0, name=f'ul_hat_1_{tp}')
        myopic.addConstr(var_ul_p[p] >= input_data.ppe_data[p[0]].expected_units + state.ul_p[p] - var_uu_p[(1, p[0])], name=f'ul_hat_2_{tp}')
        myopic.addConstr(var_ul_p[p] <= M * var_ulb[p], name=f'ul_max_1_{tp}')
        myopic.addConstr(var_ul_p[p] <= input_data.ppe_data[p[0]].expected_units + state.ul_p[p] - var_uu_p[(1, p[0])] + (M * (1-var_ulb[p])), name=f'ul_hat_2_{tp}')

    # Constraints
    # 1) Resource Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            myopic.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + state.ul_p[(tp[1],)] + var_uv[tp], name=f'resource_constraint_{tp}')
        else:
            myopic.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')

    # 2) Bounds on Reschedules
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        if ttpmdkc[0] == ttpmdkc[1] == 1:
            myopic.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')
        elif ttpmdkc[0] >= 2 and ttpmdkc[1] >= 2:
            myopic.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdkc[0], tp[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        expr.addConstant(state.ps_tmdkc[tmdkc])
        myopic.addConstr(expr >= 0, f'consistent_resc_{(tmdkc)}')
        # Scheduled
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        expr.addConstant(state.pw_mdkc[mdkc])
        myopic.addConstr(expr >= 0, f'consistent_sch_{(mdkc)}')

    # Objective Function
    # Cost Function
    def wait_cost() -> gp.LinExpr:
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()
    
        # Cost of Waiting
        for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):  
            expr.addTerms(model_param.cw[mdkc[2]], var_pw_p[mdkc])                     

        return expr
    def pref_earlier_appointment() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
            expr.addTerms(model_param.cs[tmdkc[3]][tmdkc[0]], var_sc[tmdkc])

        return expr
    def reschedule_cost() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()

        # Cost of Rescheduling                
        for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
            if ttpmdkc[0] > ttpmdkc[1]: # Good Reschedule
                difference = ttpmdkc[0] - ttpmdkc[1]
                expr.addTerms(-(model_param.cs[ttpmdkc[4]][difference] - model_param.cc[ttpmdkc[4]]), var_rsc[ttpmdkc])
            elif ttpmdkc[0] < ttpmdkc[1]: # Bad Reschedule
                difference = ttpmdkc[1] - ttpmdkc[0]
                expr.addTerms((model_param.cs[ttpmdkc[4]][difference] + model_param.cc[ttpmdkc[4]]), var_rsc[ttpmdkc])

        return expr
    def goal_violation_cost() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()

        # Modification
        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(model_param.M, var_uv[tp])

        return expr

    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost()
    pref_early = pref_earlier_appointment()
    rescheduling_cost_expr = reschedule_cost()
    goal_vio_cost_expr = goal_violation_cost()
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)
    
    myopic.setObjective(cost_expr, GRB.MINIMIZE)
    myopic.optimize()
    # print(f"\tObjective Value: {myopic.ObjVal}")
    myopic.write('myopic.lp')
    if myopic.Status != 2:
        print(state.ul_p)
        print(state.pw_mdc)
        print(state.ps_tmdc)

    # Saves Action
    sc = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        sc[tmdkc] = var_sc[tmdkc].X
    rsc = {}
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        rsc[ttpmdkc] = var_rsc[ttpmdkc].X
    uv = {}
    uvb = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uv[tp] = var_uv[tp].X
        uvb[tp] = var_uvb[tp].X

    ul_p = {}
    ulb = {}
    for p in itertools.product(indices['p']):
        ul_p[p] = var_ul_p[p].X
        ulb[p] = var_ulb[p].X
    
    uu_p = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uu_p[tp] = var_uu_p[tp].X
    pw_p = {}
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        pw_p[mdkc] = var_pw_p[mdkc].X
    ps_p = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps_p[tmdkc] = var_ps_p[tmdkc].X

    new_action = action(sc, rsc, uv, uvb, ul_p, ulb, uu_p, pw_p, ps_p)
    return new_action
def mdp_policy(input_data, state, betas) -> action:

    # Input Data
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    usage = input_data.usage
    M = input_data.model_param.M
    gamma = input_data.model_param.gamma
    arrival = input_data.arrival
    transition = input_data.transition

    # State Data
    ul = state.ul_p
    pw = state.pw_mdkc
    ps = state.ps_tmdkc

    # Initializes model
    MDP = gp.Model('MDP Policy')
    MDP.Params.LogToConsole = 0

    # Decision Variables
    var_sc = {}
    var_rsc = {}
    var_uv = {}

    var_uvb = {}
    var_ul_p = {}
    var_ulb = {}
    var_uu_p = {}
    var_pw_p = {}
    var_ps_p = {}

    # SC
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_sc[tmdkc] = MDP.addVar(name=f'a_sc_{tmdkc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_rsc[ttpmdkc] = MDP.addVar(name=f'a_rsc_{ttpmdkc}', vtype=GRB.INTEGER)
    # UV
    for tp in itertools.product(indices['t'], indices['p']):
        var_uv[tp] = MDP.addVar(name=f'a_uv_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = MDP.addVar(name=f'a_uvb{tp}', vtype=GRB.BINARY)

    # UL Hat & UL B
    for p in itertools.product(indices['p']):
        var_ul_p[p] = MDP.addVar(name=f'a_ul_p_{p}', vtype=GRB.CONTINUOUS, obj=0)
        var_ulb[p] = MDP.addVar(name=f'a_ulb_{p}', vtype=GRB.BINARY, obj=0)
    # UU Hat & UV B
    for tp in itertools.product(indices['t'], indices['p']):
        var_uu_p[tp] = MDP.addVar(name=f'a_uu_p_{tp}', vtype=GRB.CONTINUOUS)
        var_uvb[tp] = MDP.addVar(name=f'a_uvb_{tp}', vtype=GRB.BINARY)
    
    # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        var_pw_p[mdkc] = MDP.addVar(name=f'a_pw_p_{mdkc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        var_ps_p[tmdkc] = MDP.addVar(name=f'a_ps_p_{tmdkc}', vtype=GRB.INTEGER)

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdkc[1], mdkc[3])], var_ps_p[(tp[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        MDP.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdkc])
        expr.addConstant(round(-state.pw_mdkc[mdkc],0))
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        MDP.addConstr(expr == 0, name=f'pw_hat_{mdkc}')
        # PS Hat
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdkc])
        expr.addConstant(-state.ps_tmdkc[tmdkc])
        expr.addTerms(-1, var_sc[tmdkc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdkc[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
            expr.addTerms(1, var_rsc[(tmdkc[0], tp, tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        MDP.addConstr(expr == 0, name=f'ps_hat_{tmdkc}')
        # UV Maximum function
    for tp in itertools.product(indices['t'], indices['p']):
        MDP.addConstr(var_uv[tp] <= M * var_uvb[tp], name=f'uv_max_1_{tp}')
        
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        expr.addConstant(-input_data.ppe_data[tp[1]].expected_units)
        expr.addConstant(M)
        expr.addTerms(-M, var_uvb[tp])
        if tp[0] == 1:
            expr.addConstant(-state.ul_p[(tp[1],)])
        MDP.addConstr(var_uv[tp] <= expr, name=f'uv_max_2_{tp}')
        # UL Maximum function
    for p in itertools.product(indices['p']):
        MDP.addConstr(var_ul_p[p] >= 0, name=f'ul_hat_1_{tp}')
        MDP.addConstr(var_ul_p[p] >= input_data.ppe_data[p[0]].expected_units + state.ul_p[p] - var_uu_p[(1, p[0])], name=f'ul_hat_2_{tp}')
        MDP.addConstr(var_ul_p[p] <= M * var_ulb[p], name=f'ul_max_1_{tp}')
        MDP.addConstr(var_ul_p[p] <= input_data.ppe_data[p[0]].expected_units + state.ul_p[p] - var_uu_p[(1, p[0])] + (M * (1-var_ulb[p])), name=f'ul_hat_2_{tp}')

    # Constraints
    # 1) Resource Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            MDP.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + state.ul_p[(tp[1],)] + var_uv[tp], name=f'resource_constraint_{tp}')
        else:
            MDP.addConstr(var_uu_p[tp] <= input_data.ppe_data[tp[1]].expected_units + var_uv[tp], name=f'resource_constraint_{tp}')

    # 2) Bounds on Reschedules
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        if ttpmdkc[0] == ttpmdkc[1] == 1:
            MDP.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')
        elif ttpmdkc[0] >= 2 and ttpmdkc[1] >= 2:
            MDP.addConstr(var_rsc[ttpmdkc] == 0, f'resc_bound_{ttpmdkc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdkc[0], tp[0], tmdkc[1], tmdkc[2], tmdkc[3], tmdkc[4])])
        expr.addConstant(state.ps_tmdkc[tmdkc])
        MDP.addConstr(expr >= 0, f'consistent_resc_{(tmdkc)}')
        # Scheduled
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdkc[0], mdkc[1], mdkc[2], mdkc[3])])
        expr.addConstant(state.pw_mdkc[mdkc])
        MDP.addConstr(expr >= 0, f'consistent_sch_{(mdkc)}')

    # Objective Function
    # Cost Function
    def wait_cost() -> gp.LinExpr:
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()
    
        # Cost of Waiting
        for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):  
            expr.addTerms(model_param.cw[mdkc[2]], var_pw_p[mdkc])                     

        return expr
    def pref_earlier_appointment() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
            expr.addTerms(model_param.cs[tmdkc[3]][tmdkc[0]], var_sc[tmdkc])

        return expr
    def reschedule_cost() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()

        # Cost of Rescheduling                
        for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
            if ttpmdkc[0] > ttpmdkc[1]: # Good Reschedule
                difference = ttpmdkc[0] - ttpmdkc[1]
                expr.addTerms(-(model_param.cs[ttpmdkc[4]][difference] - model_param.cc[ttpmdkc[4]]), var_rsc[ttpmdkc])
            elif ttpmdkc[0] < ttpmdkc[1]: # Bad Reschedule
                difference = ttpmdkc[1] - ttpmdkc[0]
                expr.addTerms((model_param.cs[ttpmdkc[4]][difference] + model_param.cc[ttpmdkc[4]]), var_rsc[ttpmdkc])

        return expr
    def goal_violation_cost() -> gp.LinExpr:
        # Initialization
        indices = input_data.indices
        model_param = input_data.model_param
        expr = gp.LinExpr()

        # Modification
        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(model_param.M, var_uv[tp])

        return expr

    # E[V] Function
    def b0_cost() -> gp.LinExpr:
        expr = gp.LinExpr()
        expr.addConstant(betas['b0']['b_0'])
        return expr
    def b_ul_cost() -> gp.LinExpr:
        expr = gp.LinExpr() 

        for p in itertools.product(indices['p']):    
            if ppe_data[p[0]].ppe_type == 'carry-over':
                expr.addTerms(betas['ul'][p], var_ul_p[p])     

        return expr
    def b_pw_costs() -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                for k in range(len(indices['k'])):

                    mdkc = (mc[0], indices['d'][d], indices['k'][k], mc[1])

                    # When m is 0
                    if mc[0] == 0: 
                        expr.addConstant(betas['pw'][mdkc] * arrival[(mdkc[1], mdkc[2], mdkc[3])])

                    # When m is less than TL_dc
                    elif mc[0] <= (transition.wait_limit[mdkc[3]] - 1):
                        expr.addTerms(betas['pw'][mdkc], var_pw_p[(mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3])])

                    # When m is M
                    elif mc[0] == indices['m'][-1]:
                        for mm in input_data.indices['m'][-2:]:

                            expr.addTerms(betas['pw'][mdkc], var_pw_p[(mm, mdkc[1], mdkc[2], mdkc[3])])
    
                            # Complexity Change
                            tr_lim = input_data.transition.wait_limit[mdkc[3]]
                            tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                            
                            if (d != 0) & (mm >= tr_lim):
                                expr.addTerms( betas['pw'][mdkc] * tr_rate_d, var_pw_p[( mm, indices['d'][d-1], mdkc[2], mdkc[3] )] )
                                
                            if (d != (len(indices['d']) - 1)) & (mm >= tr_lim):
                                expr.addTerms(- betas['pw'][mdkc] * tr_rate_d, var_pw_p[( mm, mdkc[1], mdkc[2], mdkc[3] )] )

                            # Priority Change
                            tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                            
                            if (k != 0) & (mm >= tr_lim):
                                expr.addTerms( betas['pw'][mdkc] * tr_rate_k, var_pw_p[( mm, mdkc[1], indices['k'][k-1], mdkc[3] )] )

                            
                            if (k != (len(indices['k']) - 1)) & (mm >= tr_lim):
                                expr.addTerms(- betas['pw'][mdkc] * tr_rate_k, var_pw_p[( mm, mdkc[1], mdkc[2], mdkc[3] )] )

                    # All others
                    else:                   
                        expr.addTerms(betas['pw'][mdkc], var_pw_p[(mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3])])
 
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[mdkc[3]]
                        tr_rate_d = transition.transition_rate_comp[(mdkc[1], mdkc[3])]
                        
                        if (d != 0) & (mdkc[0]-1 >= tr_lim):
                            expr.addTerms( betas['pw'][mdkc] * tr_rate_d, var_pw_p[( mdkc[0]-1, indices['d'][d-1], mdkc[2], mdkc[3] )] )
                            
                        if (d != (len(indices['d']) - 1)) & (mdkc[0]-1 >= tr_lim):
                            expr.addTerms(- betas['pw'][mdkc] * tr_rate_d, var_pw_p[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )] )

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(mdkc[2], mdkc[3])]
                        
                        if (k != 0) & (mdkc[0]-1 >= tr_lim):
                            expr.addTerms( betas['pw'][mdkc] * tr_rate_k, var_pw_p[( mdkc[0]-1, mdkc[1], indices['k'][k-1], mdkc[3] )] )

                        
                        if (k != (len(indices['k']) - 1)) & (mdkc[0]-1 >= tr_lim):
                            expr.addTerms(- betas['pw'][mdkc] * tr_rate_k, var_pw_p[( mdkc[0]-1, mdkc[1], mdkc[2], mdkc[3] )] )

        return expr
    def b_ps_costs() -> gp.LinExpr:
        # Initialization
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                for k in range(len(indices['k'])):
                    
                    tmdkc = (tmc[0], tmc[1], indices['d'][d], indices['k'][k], tmc[2])

                    # When m = 0
                    if tmdkc[1] == 0: 
                        pass

                    # When t = T
                    elif tmdkc[0] == indices['t'][-1]:
                        pass

                    # When m is less than TL_c
                    elif tmdkc[1] < (transition.wait_limit[tmdkc[4]]):
                        expr.addTerms( betas['ps'][tmdkc], var_ps_p[((tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4]))] )

                    # When m = M
                    elif tmdkc[1] == indices['m'][-1]:

                        for mm in input_data.indices['m'][-2:]:
                            expr.addTerms(- betas['ps'][tmdkc], var_ps_p[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )] )
            
                            # Complexity Change
                            tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                            tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                            
                            if (d != 0) & (mm >= tr_lim):
                                expr.addTerms(- betas['ps'][tmdkc] * tr_rate_d, var_ps_p[( tmdkc[0]+1, mm, indices['d'][d-1], tmdkc[3], tmdkc[4] )] )
                                
                            if (d != (len(indices['d']) - 1)) & (mm >= tr_lim):
                                expr.addTerms( betas['ps'][tmdkc] * tr_rate_d, var_ps_p[ (tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4]) ] )

                            # Priority Change
                            tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                            
                            if (k != 0) & (mm >= tr_lim):
                                expr.addTerms(- betas['ps'][tmdkc] * tr_rate_k, var_ps_p[( tmdkc[0]+1, mm, tmdkc[2], indices['k'][k-1], tmdkc[4] )] )

                            
                            if (k != (len(indices['k']) - 1)) & (mm >= tr_lim):
                                expr.addTerms( betas['ps'][tmdkc] * tr_rate_k, var_ps_p[( tmdkc[0]+1, mm, tmdkc[2], tmdkc[3], tmdkc[4] )] )

                    # Everything Else
                    else:
                        expr.addTerms(- betas['ps'][tmdkc], var_ps_p[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )] )
        
                        # Complexity Change
                        tr_lim = input_data.transition.wait_limit[tmdkc[4]]
                        tr_rate_d = transition.transition_rate_comp[(tmdkc[2], tmdkc[4])]
                        
                        if (d != 0) & (tmdkc[1]-1 >= tr_lim):
                            expr.addTerms(- betas['ps'][tmdkc] * tr_rate_d, var_ps_p[( tmdkc[0]+1, tmdkc[1]-1, indices['d'][d-1], tmdkc[3], tmdkc[4] )] )
                            
                        if (d != (len(indices['d']) - 1)) & (tmdkc[1]-1 >= tr_lim):
                            expr.addTerms( betas['ps'][tmdkc] * tr_rate_d, var_ps_p[ (tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4]) ] )

                        # Priority Change
                        tr_rate_k = transition.transition_rate_pri[(tmdkc[3], tmdkc[4])]
                        
                        if (k != 0) & (tmdkc[1]-1 >= tr_lim):
                            expr.addTerms(- betas['ps'][tmdkc] * tr_rate_k, var_ps_p[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], indices['k'][k-1], tmdkc[4] )] )

                        
                        if (k != (len(indices['k']) - 1)) & (tmdkc[1]-1 >= tr_lim):
                            expr.addTerms( betas['ps'][tmdkc] * tr_rate_k, var_ps_p[( tmdkc[0]+1, tmdkc[1]-1, tmdkc[2], tmdkc[3], tmdkc[4] )] )     

        return expr

    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost()
    pref_early = pref_earlier_appointment()
    rescheduling_cost_expr = reschedule_cost()
    goal_vio_cost_expr = goal_violation_cost()
    cost_expr = gp.LinExpr(wait_cost_expr + pref_early + rescheduling_cost_expr + goal_vio_cost_expr)

            # Value
    b0_expr = b0_cost()
    b_ul_expr = b_ul_cost()
    b_pw_expr = b_pw_costs()
    b_ps_expr = b_ps_costs()
    value_expr = gp.LinExpr(b0_expr + b_ul_expr + b_pw_expr + b_ps_expr)
    
    
    MDP.setObjective(cost_expr + (gamma * value_expr), GRB.MINIMIZE)
    
    MDP.setObjective(cost_expr, GRB.MINIMIZE)
    MDP.optimize()
    MDP.write('MDP.lp')
    if MDP.Status != 2:
        print(state.ul_p)
        print(state.pw_mdc)
        print(state.ps_tmdc)

    # Saves Action
    sc = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        sc[tmdkc] = var_sc[tmdkc].X
    rsc = {}
    for ttpmdkc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        rsc[ttpmdkc] = var_rsc[ttpmdkc].X
    uv = {}
    uvb = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uv[tp] = var_uv[tp].X
        uvb[tp] = var_uvb[tp].X

    ul_p = {}
    ulb = {}
    for p in itertools.product(indices['p']):
        ul_p[p] = var_ul_p[p].X
        ulb[p] = var_ulb[p].X
    
    uu_p = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uu_p[tp] = var_uu_p[tp].X
    pw_p = {}
    for mdkc in itertools.product(indices['m'], indices['d'], indices['k'], indices['c']):
        pw_p[mdkc] = var_pw_p[mdkc].X
    ps_p = {}
    for tmdkc in itertools.product(indices['t'], indices['m'], indices['d'], indices['k'], indices['c']):
        ps_p[tmdkc] = var_ps_p[tmdkc].X

    new_action = action(sc, rsc, uv, uvb, ul_p, ulb, uu_p, pw_p, ps_p)
    return new_action

def non_zero_state(state: state):
    for key,value in state.ul_p.items():
        if value >= 0.001 or value <= -0.001: print(f'\tUnits Leftover - {key} - {state.ul_p[key]}')
    for key,value in state.pw_mdkc.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tPatients Waiting- {key} - {value}')
    for key,value in state.ps_tmdkc.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tPatients Scheduled- {key} - {value}')
def non_zero_action(action: action):
    for key,value in action.sc_tmdkc.items():
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tPatients Schedule - {key} - {value}')
    for key,value in action.rsc_ttpmdkc.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tPatients Reschedule- {key} - {value}')
    for key,value in action.uv_tp.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tUnits Violated- {key} - {value}')
    for key,value in action.ul_p_p.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tUnits Left Over - Post Decision - {key} - {value}')
    for key,value in action.uu_p_tp.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tUnits Used - Post Decision - {key} - {value}')
    for key,value in action.pw_p_mdkc.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\ttPatients Waiting - Post Decision - {key} - {value}')
    for key,value in action.ps_p_tmdkc.items(): 
        if value >= 0.001 or value <= -0.001 >= 0.1: print(f'\tPatients Scheduled - Post Decision - {key} - {value}')

def simulation(input_data, replication, days, warm_up, decision_policy, **kwargs): 

    full_data = []
    cost_data = []
    discounted_total_cost = []
    curr_state = initial_state(input_data)
        
    for repl in range(replication):
        print(f'Replication {repl+1} / {replication}')
        repl_data = []
        cost_repl_data = []
        discounted_total_cost.append(0)

        # Initializes State
        initial_state_val = deepcopy(curr_state)    

        for day in trange(days):
            # print(f'Day - {day+1}')

            # Saves Initial State Data
            # if day >= warm_up:
            repl_data.append(deepcopy(initial_state_val))

            # print('Initial State')
            # non_zero_state(initial_state_val)
            # Generate Action & Executes an Action
            
            new_action = None
            if day < warm_up:
                new_action = myopic_policy(input_data, initial_state_val)
            else:
                if 'betas' in kwargs:
                    new_action = decision_policy(input_data, initial_state_val, kwargs['betas'])
                else:
                    new_action = decision_policy(input_data, initial_state_val)
            initial_state_val = execute_action(input_data, initial_state_val, new_action)
            # print('Aciton')
            # non_zero_action(new_action)

            # Calculates cost
            cost = state_action_cost(input_data, initial_state_val, new_action)
            cost_repl_data.append(cost)
            if day >= warm_up:
                discounted_total_cost[repl] = discounted_total_cost[repl]*input_data.model_param.gamma + cost
            # print(f'Cost: {cost}')

            # Executes Transition
            initial_state_val = execute_transition(input_data, initial_state_val, new_action)

        # Save data
        full_data.append(repl_data)
        cost_data.append(cost_repl_data)
    
    return(cost_data, discounted_total_cost, full_data)
def generate_expected_values(input_data, repl, days, warmup):
    cost_data, sim_data = simulation(input_data, repl, days, warmup, myopic_policy)

    state_averages = initial_state(input_data)
    total_days = repl * (days-warmup)

    # Adjusts UE
    for key,value in state_averages.ue_tp.items():
        state_averages.ue_tp[key] = 0

    for repl_data in sim_data:
        for day_state in repl_data:

            # UE Average
            for key,value in day_state.ue_tp.items():
                state_averages.ue_tp[key] += value / total_days
            # UU Average
            for key,value in day_state.uu_tp.items():
                state_averages.uu_tp[key] += value / total_days
            # PW Average
            for key,value in day_state.pw_mdc.items():
                state_averages.pw_mdc[key] += value / total_days
            # PS Average
            for key,value in day_state.ps_tmdc.items():
                state_averages.ps_tmdc[key] += value / total_days

    # Rounds Stuff
    # UE Average
    for key,value in state_averages.ue_tp.items():
        state_averages.ue_tp[key] = round(value, 3)
    # UU Average
    for key,value in state_averages.uu_tp.items():
        state_averages.uu_tp[key] = round(value, 3)
    # PW Average
    for key,value in state_averages.pw_mdc.items():
        state_averages.pw_mdc[key] = round(value, 3)
    # PS Average
    for key,value in state_averages.ps_tmdc.items():
        state_averages.ps_tmdc[key] = round(value, 3)

    return(state_averages)