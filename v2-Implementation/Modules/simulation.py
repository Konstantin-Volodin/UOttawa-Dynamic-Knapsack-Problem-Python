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
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        if mdc[0] == 0:
            pw[mdc] = arrival[(mdc[1], mdc[2])] 
        else:
            pw[mdc] = 0
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps[tmdc] = 0

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
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = 0
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = 0
    
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
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        pw_p[mdc] = 0
    ps_p = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_p[tmdc] = 0

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
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        new_state.ps_tmdc[tmdc]= action.ps_p_tmdc[tmdc]
                
    # Patients Waiting
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        new_state.pw_mdc[mdc]= action.pw_p_mdc[mdc]

    return(new_state)
# Generates cost of state-action
def state_action_cost(input_data, state, action) -> float:
    # Initializes
    indices = input_data.indices
    model_param = input_data.model_param
    M = model_param.M
    cost = 0

    # Cost of Waiting
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):            
        cost += model_param.cw * ( action.pw_p_mdc[mdc] )

    # Cost of Waiting - Last Period
    # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
    #     cost += model_param.cw**(indices['m'][-1]+1) * ( action.ps_p_tmdc[(tdc[0],indices['m'][-1],tdc[1],tdc[2])] )

    # Cost of Later Schedulings
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        cost += (model_param.cs[tmdc[0]]) * ( action.sc_tmdc[(tmdc[0],tmdc[1],tmdc[2],tmdc[3])] )

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
        cost += action.uv_tp[tp] * M

    return(cost)
# Executes Transition to next state
def execute_transition(input_data, state, action) -> state:

    indices = input_data.indices
    new_state = deepcopy(state)

    # UL
    for p in itertools.product(indices['p']):
        deviation = np.random.uniform(
            input_data.ppe_data[p[0]].deviation[0],
            input_data.ppe_data[p[0]].deviation[1]
        )
        new_state.ul_p[p] = state.ul_p[p] + deviation

    # PW
        # Generates New Arrivals, Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == 0:
            new_state.pw_mdc[mdc] = np.random.poisson(input_data.arrival[(mdc[1], mdc[2])])
        elif mdc[0] == indices['m'][-1]:
            new_state.pw_mdc[mdc] += state.pw_mdc[(mdc[0] - 1, mdc[1], mdc[2])]
        else:
            new_state.pw_mdc[mdc] = state.pw_mdc[(mdc[0] - 1, mdc[1], mdc[2])]

        # Transitions in Difficulties
    for mc in itertools.product(indices['m'], indices['c']):
        for d in range(len(indices['d'])):
            if indices['d'][d] == indices['d'][-1]: continue
            else:
                # Transitioned Patients
                mdc = (mc[0], indices['d'][d], mc[1])
                patients_transitioned = np.random.binomial(
                    new_state.pw_mdc[mdc],
                    input_data.transition[mdc]
                )
                new_state.pw_mdc[mdc] -= patients_transitioned
                new_state.pw_mdc[(mc[0], indices['d'][d+1], mc[1])] += patients_transitioned

    # PS
        # Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for tmdc in itertools.product(indices['t'], reversed(indices['m']), indices['d'], indices['c']):
        if tmdc[0] == indices['t'][-1]:
            new_state.ps_tmdc[tmdc] = 0
        elif tmdc[1] == 0:
            new_state.ps_tmdc[tmdc] = 0
        elif tmdc[1] == indices['m'][-1]:
            new_state.ps_tmdc[tmdc] = state.ps_tmdc[(tmdc[0] + 1, tmdc[1], tmdc[2], tmdc[3])] + state.ps_tmdc[(tmdc[0] + 1, tmdc[1] - 1, tmdc[2], tmdc[3])]
        else:
            new_state.ps_tmdc[tmdc] = state.ps_tmdc[(tmdc[0] + 1, tmdc[1] - 1, tmdc[2], tmdc[3])]

        # Transitions in Difficulties
    for tmc in itertools.product(indices['t'], reversed(indices['m']), indices['c']):
        for d in range(len(indices['d'])):
            if indices['d'][d] == indices['d'][-1]: continue
            else:
                # Transitioned Patients
                tmdc = (tmc[0], tmc[1], indices['d'][d], mc[1])
                patients_transitioned = np.random.binomial(
                    new_state.ps_tmdc[tmdc],
                    input_data.transition[(tmc[1], indices['d'][d], tmc[2])]
                )
                new_state.ps_tmdc[tmdc] -= patients_transitioned
                new_state.ps_tmdc[(tmc[0], tmc[1], indices['d'][d+1], tmc[2])] += patients_transitioned

                # Change in PPE Usage
                for p in indices['p']:
                    ppe_change = input_data.usage[(p, indices['d'][d+1], tmc[2])] - input_data.usage[(p, indices['d'][d], tmc[2])]
                    new_state.uu_tp[(tmc[0], p)] += ppe_change*patients_transitioned

    return(new_state)

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
    pw = state.pw_mdc
    ps = state.ps_tmdc

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
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = myopic.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = myopic.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)
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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        var_pw_p[mdc] = myopic.addVar(name=f'a_pw_p_{mdc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps_p[tmdc] = myopic.addVar(name=f'a_ps_p_{tmdc}', vtype=GRB.INTEGER)

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdc[1], mdc[2])], var_ps_p[(tp[0], mdc[0], mdc[1], mdc[2])])
        myopic.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdc])
        expr.addConstant(round(-state.pw_mdc[mdc],0))
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdc[0], mdc[1], mdc[2])])
        myopic.addConstr(expr == 0, name=f'pw_hat_{mdc}')
        # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdc])
        expr.addConstant(-state.ps_tmdc[tmdc])
        expr.addTerms(-1, var_sc[tmdc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdc[0], tmdc[1], tmdc[2], tmdc[3])])
            expr.addTerms(1, var_rsc[(tmdc[0], tp, tmdc[1], tmdc[2], tmdc[3])])
        myopic.addConstr(expr == 0, name=f'ps_hat_{tmdc}')
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
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] == ttpmdc[1] == 1:
            myopic.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            myopic.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdc[0], tp[0], tmdc[1], tmdc[2], tmdc[3])])
        expr.addConstant(state.ps_tmdc[tmdc])
        myopic.addConstr(expr >= 0, f'consistent_resc_{(tmdc)}')
        # Scheduled
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdc[0], mdc[1], mdc[2])])
        expr.addConstant(state.pw_mdc[mdc])
        myopic.addConstr(expr >= 0, f'consistent_sch_{(mdc)}')

    # Objective Function
    # Cost Function
    def wait_cost() -> gp.LinExpr:
        expr = gp.LinExpr()
    
        # Cost of Waiting
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):  
            expr.addTerms(model_param.cw, var_pw_p[mdc])                    
        
        # Cost of Waiting - Last Period
        # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        #     expr.addTerms(model_param.cw**(indices['m'][-1]+1), var_ps_p[(tdc[0],indices['m'][-1],tdc[1],tdc[2])])     

        return expr
    def pref_earlier_appointment() -> gp.LinExpr:
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(model_param.cs[tmdc[0]], var_sc[tmdc])
        return expr
    def reschedule_cost() -> gp.LinExpr:

        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[0] > ttpmdc[1]: # good reschedule
                difference = ttpmdc[0] - ttpmdc[1]
                expr.addTerms(-(model_param.cs[difference] - model_param.cc), var_rsc[ttpmdc])
            elif ttpmdc[1] > ttpmdc[0]: # bad reschedule
                difference = ttpmdc[1] - ttpmdc[0]
                expr.addTerms(model_param.cs[difference] + model_param.cc, var_rsc[ttpmdc])

        return expr
    def goal_violation_cost() -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var_uv[tp])

        return(expr)
    
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
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = var_sc[tmdc].X
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = var_rsc[ttpmdc].X
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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        pw_p[mdc] = var_pw_p[mdc].X
    ps_p = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_p[tmdc] = var_ps_p[tmdc].X

    new_action = action(sc, rsc, uv, uvb, ul_p, ulb, uu_p, pw_p, ps_p)
    return new_action
def mdp_policy(input_data, state, betas) -> action:

    # Input Data
    indices = input_data.indices
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    usage = input_data.usage
    M = input_data.model_param.M
    gamma = model_param.gamma
    transition = input_data.transition
    arrival = input_data.arrival

    # State Data
    ul = state.ul_p
    pw = state.pw_mdc
    ps = state.ps_tmdc

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
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = MDP.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = MDP.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)
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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        var_pw_p[mdc] = MDP.addVar(name=f'a_pw_p_{mdc}', vtype=GRB.INTEGER)
    # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps_p[tmdc] = MDP.addVar(name=f'a_ps_p_{tmdc}', vtype=GRB.INTEGER)

    # Auxiliary Variable Definition
        # UU Hat
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_uu_p[tp])
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1], mdc[1], mdc[2])], var_ps_p[(tp[0], mdc[0], mdc[1], mdc[2])])
        MDP.addConstr(expr == 0, name=f'uu_hat_{tp}')
        # PW Hat
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_pw_p[mdc])
        expr.addConstant(round(-state.pw_mdc[mdc],0))
        for t in indices['t']:
            expr.addTerms(1, var_sc[(t, mdc[0], mdc[1], mdc[2])])
        MDP.addConstr(expr == 0, name=f'pw_hat_{mdc}')
        # PS Hat
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        expr.addTerms(1, var_ps_p[tmdc])
        expr.addConstant(-state.ps_tmdc[tmdc])
        expr.addTerms(-1, var_sc[tmdc])
        for tp in indices['t']:
            expr.addTerms(-1, var_rsc[(tp, tmdc[0], tmdc[1], tmdc[2], tmdc[3])])
            expr.addTerms(1, var_rsc[(tmdc[0], tp, tmdc[1], tmdc[2], tmdc[3])])
        MDP.addConstr(expr == 0, name=f'ps_hat_{tmdc}')
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
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] == ttpmdc[1] == 1:
            MDP.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            MDP.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 3) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdc[0], tp[0], tmdc[1], tmdc[2], tmdc[3])])
        expr.addConstant(state.ps_tmdc[tmdc])
        MDP.addConstr(expr >= 0, f'consistent_resc_{(tmdc)}')
        # Scheduled
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdc[0], mdc[1], mdc[2])])
        expr.addConstant(state.pw_mdc[mdc])
        MDP.addConstr(expr >= 0, f'consistent_sch_{(mdc)}')

    # Objective Function
    # Cost Function
    def wait_cost() -> gp.LinExpr:
        expr = gp.LinExpr()
    
        # Cost of Waiting
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):  
            expr.addTerms(model_param.cw, var_pw_p[mdc])                    
        
        # Cost of Waiting - Last Period
        # for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        #     expr.addTerms(model_param.cw**(indices['m'][-1]+1), var_ps_p[(tdc[0],indices['m'][-1],tdc[1],tdc[2])])     

        return expr
    def pref_earlier_appointment() -> gp.LinExpr:
        expr = gp.LinExpr()
        
        # Prefer Earlier Appointments
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(model_param.cs[tmdc[0]], var_sc[tmdc])
        return expr
    def reschedule_cost() -> gp.LinExpr:

        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[0] > ttpmdc[1]: # good reschedule
                difference = ttpmdc[0] - ttpmdc[1]
                expr.addTerms(-(model_param.cs[difference] - model_param.cc), var_rsc[ttpmdc])
            elif ttpmdc[1] > ttpmdc[0]: # bad reschedule
                difference = ttpmdc[1] - ttpmdc[0]
                expr.addTerms(model_param.cs[difference] + model_param.cc, var_rsc[ttpmdc])


        return expr
    def goal_violation_cost() -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addTerms(M, var_uv[tp])

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
                expr.addConstant(betas['ul'][p[0]] * state.ul_p[p])
                expr.addTerms(- (betas['ul'][p[0]] * gamma), var_ul_p[p])     
            elif ppe_data[p[0]].ppe_type == 'non-carry-over':
                expr.addConstant(betas['ul'][p[0]] * state.ul_p[p])

        return expr
    def b_pw_costs() -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])

                # When m is 0
                if mc[0] == 0: 
                    expr.addConstant(betas['pw'][mdc] * gamma * arrival[mdc[1], mdc[2]])

                # When m is M
                elif mc[0] == indices['m'][-1]:

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(betas['pw'][mdc] * gamma, var_pw_p[(mm, mdc[1], mdc[2])])
           
                        # Transitioned In
                        if d != 0:
                            expr.addTerms(
                                betas['pw'][mdc] * gamma * transition[( mm, indices['d'][d-1], mdc[2] )],
                                var_pw_p[( mm, indices['d'][d-1], mdc[2] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                -betas['pw'][mdc] * gamma * transition[( mm, mdc[1], mdc[2] )],
                                var_pw_p[( mm, mdc[1], mdc[2] )]
                            )

                # All others
                else:                   
                    expr.addTerms(betas['pw'][mdc] * gamma, var_pw_p[(mdc[0]-1, mdc[1], mdc[2])])
           
                    # Transitioned In
                    if d != 0:
                        expr.addTerms(
                            betas['pw'][mdc] * gamma * transition[( mdc[0]-1, indices['d'][d-1], mdc[2] )],
                            var_pw_p[( mdc[0]-1, indices['d'][d-1], mdc[2] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            -betas['pw'][mdc] * gamma * transition[( mdc[0]-1, mdc[1], mdc[2] )],
                            var_pw_p[( mdc[0]-1, mdc[1], mdc[2] )]
                        )

        return(expr)
    def b_ps_costs() -> gp.LinExpr:
        expr = gp.LinExpr()

        for tmc in itertools.product(indices['t'], indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                tmdc = (tmc[0], tmc[1], indices['d'][d], tmc[2])

                # When m is 0
                if tmdc[1] == 0: 
                    pass

                # When t is T
                elif tmdc[0] == indices['t'][-1]:
                    pass

                # when m is M
                elif tmdc[1] == indices['m'][-1]:

                    for mm in input_data.indices['m'][-2:]:
                        expr.addTerms(betas['ps'][tmdc] * gamma, var_ps_p[(tmdc[0]+1, mm, tmdc[2], tmdc[3])])
           
                        # Transitioned In
                        if d != 0:
                            expr.addTerms(
                                betas['ps'][tmdc] * gamma * transition[( mm, indices['d'][d-1], tmdc[3] )],
                                var_ps_p[( tmdc[0]+1, mm, indices['d'][d-1], tmdc[3] )]
                            )
                        # Transitioned Out
                        if d != indices['d'][-1]:
                            expr.addTerms(
                                -betas['ps'][tmdc] * gamma * transition[( mm, tmdc[2], tmdc[3] )],
                                var_ps_p[( tmdc[0]+ 1, mm, tmdc[2], tmdc[3] )]
                            )
                
                # Everything Else
                else:
                    expr.addTerms(betas['ps'][tmdc] * gamma, var_ps_p[(tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3])])
           
                    # Transitioned In
                    if d != 0:
                        expr.addTerms(
                            betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )],
                            var_ps_p[( tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3] )]
                        )
                    # Transitioned Out
                    if d != indices['d'][-1]:
                        expr.addTerms(
                            -betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, tmdc[2], tmdc[3] )],
                            var_ps_p[( tmdc[0]+ 1, tmdc[1]-1, tmdc[2], tmdc[3] )]
                        )

        return(expr)
    

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
    MDP.optimize()
    MDP.write('mdp.lp')
    if MDP.Status != 2:
        MDP.computeIIS()
        MDP.write('MDP.ilp')

        print(state.ul_p)
        print()
        print(state.pw_mdc)
        print()
        print(state.ps_tmdc)
        print()

    # Saves Action
    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = var_sc[tmdc].X
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = var_rsc[ttpmdc].X
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
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        pw_p[mdc] = var_pw_p[mdc].X
    ps_p = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps_p[tmdc] = var_ps_p[tmdc].X

    new_action = action(sc, rsc, uv, uvb, ul_p, ulb, uu_p, pw_p, ps_p)
    return new_action

def non_zero_state(state: state):
    for key,value in state.ul_p.items():
        if state.ul_p[key] >= 0.1: print(f'\tUnits Leftover - {key} - {state.ul_p[key]}')
    for key,value in state.pw_mdc.items(): 
        if value >= 0.1: print(f'\tPatients Waiting- {key} - {value}')
    for key,value in state.ps_tmdc.items(): 
        if value >= 0.1: print(f'\tPatients Scheduled- {key} - {value}')
def non_zero_action(action: action):
    for key,value in action.sc_tmdc.items():
        if value >= 0.1: print(f'\tPatients Schedule - {key} - {value}')
    for key,value in action.rsc_ttpmdc.items(): 
        if value >= 0.1: print(f'\tPatients Reschedule- {key} - {value}')
    for key,value in action.uv_tp.items(): 
        if value >= 0.1: print(f'\tUnits Violated- {key} - {value}')
    for key,value in action.ul_p_p.items(): 
        if value >= 0.1: print(f'\tUnits Left Over - Post Decision - {key} - {value}')
    for key,value in action.uu_p_tp.items(): 
        if value >= 0.1: print(f'\tUnits Used - Post Decision - {key} - {value}')
    for key,value in action.pw_p_mdc.items(): 
        if value >= 0.1: print(f'\ttPatients Waiting - Post Decision - {key} - {value}')
    for key,value in action.ps_p_tmdc.items(): 
        if value >= 0.1: print(f'\tPatients Scheduled - Post Decision - {key} - {value}')

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