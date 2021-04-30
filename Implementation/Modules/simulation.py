from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from Modules.data_classes import state, action, variables

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
        pw[mdc] = 0
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        ps[tmdc] = 0

    # Unit States
    ue = {}
    uu = {}
    uv = {}
    for tp in itertools.product(indices['t'], indices['p']):
        ue[tp] = ppe_data[tp[1]].expected_units
        uv[tp] = 0
        uu[tp] = 0

    init_state = state(ue, uu, uv, pw, ps)

    return init_state
# Initializes Action ( Everything Empty )
def initial_action(input_data) -> action:
    indices = input_data.indices

    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = 0
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = 0

    # Returns
    init_action = action(sc, rsc)
    return(init_action)

# Executes Action
def execute_action(input_data, state, action) -> state:
    
    new_state = deepcopy(state)

    # Handles Reschedules
    for t in input_data.indices['t']:
        for tp in input_data.indices['t']:
            for mdc in itertools.product(input_data.indices['m'], input_data.indices['d'], input_data.indices['c']):

                # Change
                rescheduled_number = action.rsc_ttpmdc[(t, tp, mdc[0], mdc[1], mdc[2])]

                # Reschedules Out
                new_state.ps_tmdc[(t, mdc[0], mdc[1], mdc[2])] -= rescheduled_number
                # Schedules into
                new_state.ps_tmdc[(tp, mdc[0], mdc[1], mdc[2])] += rescheduled_number
                
    # Handles Schedules
    for m in input_data.indices['m']:
        for d in input_data.indices['d']:
            for c in input_data.indices['c']:

                # Adjusts PS
                total_sched = 0
                for t in input_data.indices['t']:
                    new_state.ps_tmdc[(t,m,d,c)] += action.sc_tmdc[(t, m, d, c)]
                    total_sched += action.sc_tmdc[(t, m, d, c)]

                # Adjusts PW
                new_state.pw_mdc[(m,d,c)] -= total_sched

                # Adjusts Resource Usage
                for t in input_data.indices['t']:
                    for p in input_data.indices['p']:
                        new_state.uu_tp[t, p] += input_data.usage[(p, d, c)] * action.sc_tmdc[(t, m, d, c)]
    


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
        cost += model_param.cw**mdc[0] * ( state.pw_mdc[mdc] )

    # Cost of Waiting - Last Period
    for tdc in itertools.product(indices['t'], indices['d'], indices['c']):
        cost += model_param.cw**indices['m'][-1] * ( state.ps_tmdc[(tdc[0],indices['m'][-1],tdc[1],tdc[2])] )

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
# Executes Transition to next state
def execute_transition(input_data, state) -> state:

    indices = input_data.indices
    new_state = deepcopy(state)

    # UE
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == 1:
            deviation = np.random.uniform(
                input_data.ppe_data[tp[1]].deviation[0],
                input_data.ppe_data[tp[1]].deviation[1]
            )
            new_state.ue_tp[tp] = state.ue_tp[(tp[0]+1, tp[1])] + deviation
        elif tp[0] == indices['t'][-1]:
            new_state.ue_tp[tp] = input_data.ppe_data[tp[1]].expected_units
        else:
            new_state.ue_tp[tp] = state.ue_tp[(tp[0]+1, tp[1])]


    # UU
    for tp in itertools.product(indices['t'], indices['p']):
        if tp[0] == indices['t'][-1]:
            new_state.uu_tp[tp] = 0
        else:
            new_state.uu_tp[tp] = state.uu_tp[(tp[0]+1, tp[1])]

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
                    new_state.pw[mdc],
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
                    new_state.ps[tmdc],
                    input_data.transition[(tmc[1], indices['d'][d], tmc[2])]
                )
                new_state.ps_tmdc[tmdc] -= patients_transitioned
                new_state.ps_tmdc[(tmc[0], tmc[1], indices['d'][d+1], tmc[2])] += patients_transitioned

                # Change in PPE Usage
                for p in indices['p']:
                    ppe_change = input_data.usage[(p, indices['d'][d+1], tmc[2])] - input_data.usage[(p, indices['d'][d], tmc[2])]
                    new_state.uu_tp[(tmc[0], p)] += ppe_change*patients_transitioned

    # UV
    for tp in itertools.product(indices['t'], indices['p']):
        new_state.uv_tp[tp] = np.max([0, new_state.uu_tp[tp] - new_state.ue_tp[tp]])

    return(new_state)

# Various Policies
def fas_policy(input_data, state) -> action:
    init_action = initial_action(input_data)

    # Retrieves capacity
    capacity = state.ue_tp.copy()
    for tp in capacity.keys():
        capacity[tp] = capacity[tp] - state.uu_tp[tp]

    # Reschedules out of day 1 if necessary
    for mdc in itertools.product(input_data.indices['m'], input_data.indices['d'],input_data.indices['c']):
        t = 1

        # Reschedules out of period 1 until there is no violation
        need_to_reschedule = False

        while True:

            # Check if capacity is violated
            need_to_reschedule = False
            # print(capacity)
            for p in input_data.indices['p']:
                if capacity[(t, p)] <= 0:
                    need_to_reschedule = True


            if need_to_reschedule == False:
                break

            # Finds slot to reschedule to and adjusts capacity metrics
            for tp in input_data.indices['t']:
                enough_capacity = False
                for p in input_data.indices['p']:
                    if capacity[(tp, p)] >= input_data.usage[(p, mdc[1], mdc[2])]:
                        enough_capacity = True

                if enough_capacity:

                    init_action.rsc_ttpmdc[(1, tp, mdc[0], mdc[1], mdc[2])] += 1
                    for p in input_data.indices['p']:
                        capacity[(t,p)] += input_data.usage[(p, mdc[1], mdc[2])]
                        capacity[(tp,p)] -= input_data.usage[(p, mdc[1], mdc[2])]
                    break
        
        if need_to_reschedule == False:
            break

    # Schedules patients starting from those who waited the longest
    for m in reversed(input_data.indices['m']): 
        for dc in itertools.product(reversed(input_data.indices['d']), input_data.indices['c']):
            mdc = (m, dc[0], dc[1])

            # Extracts number of people to schedule and usage of this patient type
            patients_to_schedule = state.pw_mdc[mdc]
            patients_scheduled = 0
            usage = {}
            for p in input_data.indices['p']:
                usage[p] = input_data.usage[(p, dc[0], dc[1])]

            # Scheduling Action
            for patient in range(patients_to_schedule):

                if patients_to_schedule == patients_scheduled: break

                for t in input_data.indices['t']:

                    # Checks Capacity
                    available_capacity = True
                    for p in input_data.indices['p']:
                        if usage[p] > capacity[(t, p)]:
                            available_capacity = False

                    # Schedules if there is capacity
                    if available_capacity == False: 
                        continue 
                    
                    init_action.sc_tmdc[(t, m, dc[0], dc[1])] += 1
                    for p in input_data.indices['p']:
                        capacity[(t, p)] -= usage[p]
                        patients_scheduled += 1
                    
                    break

    # Schedules into day 1 if it can
    #               
    return(init_action)
def myopic_policy(input_data, state) -> action:
    # Input Data
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # State Data
    ue = state.ue_tp
    uu = state.uu_tp
    uv = state.uv_tp
    pw = state.pw_mdc
    ps = state.ps_tmdc

    # Initializes model
    myopic = gp.Model('Myopic Policy')
    # myopic.Params.LogToConsole = 0

    # Decision Variables
    var_sc = {}
    var_rsc = {}

    # Actions
    # SC
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = myopic.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = myopic.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)

    # Constraints
    # 1) PPE Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        # Patients scheduled - action
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(usage[(tp[1],mdc[1],mdc[2])], var_sc[(tp[0], mdc[0],mdc[1],mdc[2])])
        # Patients rescheduled into
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(usage[(tp[1],tmdc[2],tmdc[3])], var_rsc[(tmdc[0], tp[0], tmdc[1],tmdc[2],tmdc[3])])
        # Patients rescheduled out
        myopic.addConstr(expr <=  ue[tp] - uu[tp] + uv[tp], name=f'tracks_usage{tp}')

    # 3) Bounds on Reschedules
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] == ttpmdc[1]:
            myopic.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            myopic.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] == 1 and ttpmdc[1] >= 3:
            myopic.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 4) Number of people schedules/reschedules must be consistent
        # Reschedules
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for tp in itertools.product(indices['t']):
            expr.addTerms(-1, var_rsc[(tmdc[0], tp[0], tmdc[1], tmdc[2], tmdc[3])])
        expr.addConstant(1 * ps[tmdc])
        myopic.addConstr(expr >= 0, f'consistent_resc_{(tmdc)}')
        # Scheduled
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        expr = gp.LinExpr()
        for t in itertools.product(indices['t']):
            expr.addTerms(-1, var_sc[(t[0], mdc[0], mdc[1], mdc[2])])
        expr.addConstant(1 * pw[mdc])
        myopic.addConstr(expr >= 0, f'consistent_sch_{(mdc)}')

    # Objective Function
    # Cost Function
    def wait_cost() -> gp.LinExpr:
        expr = gp.LinExpr()
    
        # PW Cost
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']): 
            expr.addConstant((model_param.cw**mdc[0]) * pw[mdc])
            for t in indices['t']:
                expr.addTerms(-(model_param.cw**mdc[0]), var_sc[(t, mdc[0], mdc[1], mdc[2])])
        # PS Cost
        for tdc in itertools.product(indices['t'], indices['d'], indices['c']): 
            expr.addConstant((model_param.cw**indices['m'][-1]) * ps[(tdc[0], indices['m'][-1], tdc[1], tdc[2])])

        return(expr)
    def reschedule_cost() -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[1] > ttpmdc[0]:
                expr.addTerms(model_param.cc, var_rsc[ttpmdc])
            elif ttpmdc[1] < ttpmdc[0]:
                expr.addTerms(-model_param.cc, var_rsc[ttpmdc])

        return(expr)
    def goal_violation_cost() -> gp.LinExpr:
        expr = gp.LinExpr()

        for tp in itertools.product(indices['t'], indices['p']):
            expr.addConstant(M * uv[tp])

        return(expr)
    
    # Generates Objective Function
        # Cost
    wait_cost_expr = wait_cost()
    rescheduling_cost_expr = reschedule_cost()
    goal_vio_cost_expr = goal_violation_cost()
    cost_expr = gp.LinExpr(wait_cost_expr + rescheduling_cost_expr + goal_vio_cost_expr)
    
    myopic.setObjective(cost_expr, GRB.MINIMIZE)
    myopic.optimize()
    myopic.write('myopic.lp')

    # Saves Action
    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = var_sc[tmdc].X
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        rsc[ttpmdc] = var_rsc[ttpmdc].X
    new_action = action(sc, rsc)
    
    return(new_action)

def mdp_policy(input_data, state, betas) -> action:
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # Initializes model
    sub_model = gp.Model('MDD Policy')

    # Decision Variables
    var_sc = {}
    var_rsc = {}

    # Actions
    # SC
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = sub_model.addVar(name=f'a_sc_{tmdc}', vtype=GRB.INTEGER)
    # RSC
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = sub_model.addVar(name=f'a_rsc_{ttpmdc}', vtype=GRB.INTEGER)

    sub_vars = variables( var_ue, var_uu, var_uv, var_pw, var_ps, var_sc, var_rsc)

    # Constraints
    # 1) PPE Usage Constraint
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        # Patients scheduled - state
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1],mdc[1],mdc[2])], var_ps[(tp[0], mdc[0],mdc[1],mdc[2])])
        # Patients scheduled - action
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1],mdc[1],mdc[2])], var_sc[(tp[0], mdc[0],mdc[1],mdc[2])])
        # Patients rescheduled
        for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
            expr.addTerms(-usage[(tp[1],tmdc[2],tmdc[3])], var_rsc[(tmdc[0], tp[0], tmdc[1],tmdc[2],tmdc[3])])
        expr.addTerms(1, var_uu[tp])
        sub_model.addConstr(expr == 0, name=f'tracks_usage{tp}')


    # 2) PPE Capacity
    for tp in itertools.product(indices['t'], indices['p']):
        expr = gp.LinExpr()
        expr.addTerms(-1, var_ue[tp])
        expr.addTerms(1, var_uu[tp])
        expr.addTerms(-1, var_uv[tp])
        sub_model.addConstr(expr <= 0, name=f'ppe_capacity_constraint_{tp}')

    # 3) Bounds on Reschedules
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[0] == ttpmdc[1]:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] >= 2 and ttpmdc[1] >= 2:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')
        elif ttpmdc[0] == 1 and ttpmdc[1] >= 3:
            sub_model.addConstr(var_rsc[ttpmdc] == 0, f'resc_bound_{ttpmdc}')

    # 4) Cap on max schedule/reschedule wait time

    # 5) Number of people schedules/reschedules must be consistent
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
    
        # PW Cost
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']): 
            expr.addTerms(model_param.cw**mdc[0], var.s_pw[mdc])
        # PS Cost
        for tdc in itertools.product(indices['t'], indices['d'], indices['c']): 
            expr.addTerms(model_param.cw**indices['m'][-1], var.s_ps[(tdc[0], indices['m'][-1], tdc[1], tdc[2])])

        return(expr)
    def reschedule_cost(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
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
                        expr.addTerms( usage[(tp[1], dc[0], dc[1])], var.a_rsc[(tp[0],tpm[0], tpm[1], dc[0], dc[1])] )
                    for tm in itertools.product(indices['t'], indices['m']):
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
                            -betas['uu'][tp] * gamma * transition[(mc[0], indices['d'][d], mc[1])] * (change_in_usage), 
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
                    # if tmdc[1] == 0: continue
                    expr.addTerms( 
                        betas['uu'][tp] * gamma * usage[(tp[1], mdc[1], mdc[2])] , 
                        var.a_rsc[(tmdc[0]+1, tp[0], tmdc[1], tmdc[2], tmdc[3])] 
                    )
                for tpmdc in itertools.product(indices['t'], indices['m'],indices['d'],indices['c']):
                    if not (tpmdc[0]+1 in indices['t']): continue
                    # if tpmdc[1] == 0: continue
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
    def b_pw_costs(var: variables, betas) -> gp.LinExpr:
        expr = gp.LinExpr()

        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])

                # When m = 0
                if mc[0] == 0: 
                    expr.addTerms(betas['pw'][mdc], var.s_pw[mdc])
                    expr.addConstant(-betas['pw'][mdc] * gamma * arrival[mdc[1], mdc[2]])

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

                # When t is T or m is 0
                if (tmdc[0] == indices['t'][-1]) or (tmdc[1] == 0):
                    expr.addTerms( betas['ps'][tmdc], var.s_ps[tmdc] )

                # When m is M
                elif tmdc[1] == indices['m'][-1]:
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

                # All others
                else: 
                    # Baseline
                    expr.addTerms( betas['ps'][tmdc], var.s_ps[tmdc] )
                    # Transition in difficulties
                    if tmdc[1] >= 1:
                        expr.addTerms( -betas['ps'][tmdc]*gamma * (1 - transition[( tmdc[1]-1, tmdc[2], tmdc[3] )]), var.s_ps[ ( tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3] ) ] )
                        if d != 0: 
                            expr.addTerms( -betas['ps'][tmdc] * gamma * transition[( tmdc[1]-1, indices['d'][d-1], tmdc[3] )], var.s_ps[ (tmdc[0]+1, tmdc[1]-1, indices['d'][d-1], tmdc[3]) ] )
                    
                    # Scheduling / Rescheduling
                    expr.addTerms( betas['ps'][tmdc] * gamma, var.a_sc[ (tmdc[0]+1, tmdc[1]-1, tmdc[2], tmdc[3]) ] )
                    if tmdc[1] >= 1:
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
    b_pw_expr = b_pw_costs(sub_vars, betas)
    b_ps_expr = b_ps_costs(sub_vars, betas)
    value_expr = gp.LinExpr(b0_expr + b_ue_expr + b_uu_expr + b_uv_expr + b_pw_expr + b_ps_expr)

    if phase1:
        sub_model.setObjective(-value_expr, GRB.MINIMIZE)
    else:
        sub_model.setObjective(cost_expr - value_expr, GRB.MINIMIZE)

    return sub_model, sub_vars




def simulation(input_data, replication, days, warm_up, decision_policy, save_data = False): 

    full_data = []
    cost_data = []

    for repl in trange(replication):
        repl_data = []
        cost_repl_data = []

        # Initializes State
        curr_state = initial_state(input_data)

        for day in range(days):

            # Saves Initial State Data
            if save_data:
                if day >= warm_up:
                    repl_data.append(deepcopy(curr_state))

            # Generate Action & Executes an Action
            new_action = decision_policy(input_data, curr_state)
            curr_state = execute_action(input_data, curr_state, new_action)

            # Calculates cost
            cost = state_action_cost(input_data, curr_state, new_action)
            if day >= warm_up:
                cost_repl_data.append(cost)

            # Executes Transition
            curr_state = execute_transition(input_data, curr_state)

        # Save data
        if save_data:
            full_data.append(repl_data)
        cost_data.append(cost_repl_data)
    
    if save_data:
        return(full_data)
    else:
        return(cost_data)
def generate_expected_values(input_data, repl, days):
    sim_data = simulation(input_data, repl, days, 300, fas_policy, True)

    state_averages = initial_state(input_data)
    total_days = repl * (days-300)

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
            # UV Average
            for key,value in day_state.uv_tp.items():
                state_averages.uv_tp[key] += value / total_days
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
    # UV Average
    for key,value in state_averages.uv_tp.items():
        state_averages.uv_tp[key] = round(value, 3)
    # PW Average
    for key,value in state_averages.pw_mdc.items():
        state_averages.pw_mdc[key] = round(value, 3)
    # PS Average
    for key,value in state_averages.ps_tmdc.items():
        state_averages.ps_tmdc[key] = round(value, 3)

    return(state_averages)