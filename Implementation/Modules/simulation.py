# %% 
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from Modules.data_classes import state, action, variables

from copy import deepcopy
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# %%

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
    # PW
        # Generates New Arrivals, Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for mdc in itertools.product(reversed(indices['m']), indices['d'], indices['c']):
        if mdc[0] == 0:
            new_state.pw_mdc[mdc] = np.random.poisson(input_data.arrival[(mdc[1], mdc[2])])
        elif mdc[0] == indices['m'][-1]:
            new_state.pw_mdc[mdc] += new_state.pw_mdc[(mdc[0] - 1, mdc[1], mdc[2])]
        else:
            new_state.pw_mdc[mdc] = new_state.pw_mdc[(mdc[0] - 1, mdc[1], mdc[2])]
        # Transitions in Difficulties
    for mc in itertools.product(reversed(indices['m']), indices['c']):
        for d in range(len(indices['d'])):
            if mc[0] == 0: continue
            elif indices['d'][d] == indices['d'][-1]: continue
            else:
                mdc = (mc[0], indices['d'][d], mc[1])
                patients_transitioned = np.random.binomial(
                    new_state.pw[mdc],
                    input_data.transition[mdc]
                )
                new_state.pw_mdc[mdc] -= patients_transitioned
                new_state.pew_mdc[(mc[0], indices['d'][d+1], mc[1])] += patients_transitioned

    # PS
        # Shifts Everyone by 1 Month, Accumulates those who waited past limit
    for tmdc in itertools.product(indices['t'], reversed(indices['m']), indices['d'], indices['c']):
        if tmdc[0] == indices['t'][-1]:
            new_state.ps_tmdc[tmdc] = 0
        elif tmdc[1] == indices['m'][-1]:
            new_state.ps_tmdc[tmdc] = new_state.ps_tmdc[(tmdc[0] + 1, tmdc[1], tmdc[2], tmdc[3])] + new_state.ps_tmdc[(tmdc[0] + 1, tmdc[1] - 1, tmdc[2], tmdc[3])]
        else:
            new_state.ps_tmdc[tmdc] = new_state.ps_tmdc[(tmdc[0] + 1, tmdc[1] - 1, tmdc[2], tmdc[3])]
        # Transitions in Difficulties
    # UE

    # UU

    # UV

    return(new_state)

# Various Policies
def fas_policy(input_data, state) -> action:
    init_action = initial_action(input_data)
    # Reschedules out of day 1 if necessary

    # Retrieves capacity
    capacity = state.ue_tp.copy()
    for tp in capacity.keys():
        capacity[tp] = capacity[tp] - state.uu_tp[tp]

    # Schedules patients starting from those who waited the longest
    for m in reversed(input_data.indices['m']): 
        for dc in itertools.product(input_data.indices['d'], input_data.indices['c']):
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
                      
    return(init_action)
def myopic_policy(input_data, state) -> action:
    pass
def mdp_policy(input_data, state, betas) -> action:
    pass

def simulation(input_data, replication, days, decision_policy): 

    full_data = []

    for repl in range(replications):
        
        repl_data = []

        # Initializes State
        curr_state = initial_state(input_data)

        for day in range(days):

            # Saves Initial State Data

            # Generate Action & Executes an Action
            new_action = decision_policy(input_data, curr_state)
            curr_state = execute_action(input_data, curr_state, new_action)

            # Calculates cost
            cost = state_action_cost(curr_state, new_action)
            print(cost)

            # Transition
            curr_state = transition(input_data, state, action)

            # Generate Arrivals
            arrivals = {}
            for dc, val in input_data.arrival.items():
                arrivals[dc] = np.random.poisson(val)
            # Assign Arrivals
            pass

            # Save data
            repl_data.append(day)

        # Save data
        full_data.append(repl)

        # print(full_data)
        


# %%
curr_state = initial_state(input_data)
# for key in curr_state.pw_mdc.keys():
#     curr_state.pw_mdc[key] = 2

new_action = fas_policy(input_data, curr_state)
curr_state = execute_action(input_data, curr_state, new_action)
curr_state = execute_transition(input_data, curr_state)
# state_action_cost(input_data, curr_state, new_action)
# %%
