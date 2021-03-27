# %%
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable


import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# %%
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


def generate_initial_state_action(input_data):
    # Input Data
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data

    # Patient States
    pe = {}
    for dc in itertools.product(indices['d'], indices['c']):
        pe[dc] = arrival[dc]
    pw = {}
    for mdc in itertools.product(indices['m'],indices['d'], indices['c']):
        if mdc[0] == 0: continue
        pw[mdc] = 0
    ps = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        if tmdc[1] == 0: continue
        ps[tmdc] = 0

    # Unit States
    ue = {}
    for tp in itertools.product(indices['t'], indices['p']):
        expected_units = ppe_data[tp[1]].expected_units
        ue[tp] = expected_units
    uu = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uu[tp] = 0
    uv = {}
    for tp in itertools.product(indices['t'], indices['p']):
        uv[tp] = np.max((0, uu[tp] - ue[tp]))

    # Actions
    sc = {}
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        sc[tmdc] = 0
        if tmdc[0] == 1 & tmdc[1] == 0:
            sc[tmdc] = arrival[(tmdc[2], tmdc[3])]
    rsc = {}
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        if ttpmdc[2] == 0: continue
        if ttpmdc[0] != 1: rsc[ttpmdc] = 0
        elif ttpmdc[1] != 1: rsc[ttpmdc] = 0
        else: rsc[ttpmdc] = 0

    # test_state = state(ue, uu, uv, pe, pw, ps)
    test_action = action(sc, rsc)
    test_state = state(
        input_data.expected_state_values['ue'],
        input_data.expected_state_values['uu'],
        input_data.expected_state_values['uv'],
        input_data.expected_state_values['pe'],
        input_data.expected_state_values['pw'],
        input_data.expected_state_values['ps']
    )
    return test_state, test_action

# Generate Phase 1 Master Model (finding initial feasible solution)
def generate_phase1_master_model(input_data,state_action_data):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma
    
    mast_model, variables, constraints = generate_master_model(input_data, state_action_data)
    
    # Goal variables
    gv_0 = mast_model.addVar(name=f'gv_beta_0')
    gv_ue = []
    gv_uu = []
    gv_uv = []
    gv_pw = []
    gv_pe = []
    gv_ps = []
    for tp in itertools.product(indices['t'], indices['p']):
        gv_ue.append(mast_model.addVar(name=f'gv_beta_ue_tp_{tp}'))
        gv_uu.append(mast_model.addVar(name=f'gv_beta_uu_tp_{tp}'))
        gv_uv.append(mast_model.addVar(name=f'gv_beta_uv_tp_{tp}'))
    for dc in itertools.product(indices['d'], indices['d']):
        gv_pe.append(mast_model.addVar(name=f'gv_beta_pe_dc_{dc}'))
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == 0: continue
        gv_pw.append(mast_model.addVar(name=f'gv_beta_pw_mdc_{mdc}'))
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        gv_ps.append(mast_model.addVar(name=f'gv_beta_ps_tmdc_{tmdc}'))

    # Modifies Constraints to include goal variables

    # New Objective Function 

    return mast_model, variables, constraints

# Generates Master model (finding optimal solution)
def generate_master_model(input_data, state_action_data):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # Cost Function
    def cost_function(state: state, action: action) -> constraint_parameter:
        cost = 0

        # Cost of Waiting
        for mdc in itertools.product(indices['m'], indices['d'], indices['c']):

            # sum_t sc part
            psc = 0 
            for t in indices['t']:
                psc += action.sc_tmdc[(t, mdc[0], mdc[1], mdc[2])]
            
            # if m = 0, uses unscheduled patients who just arrived
            # if m > 0, uses patients waiting for m
            if mdc[0] == 0: cost += model_param.cw**mdc[0] * ( state.pe_dc[(mdc[1],mdc[2])] - psc )
            else: cost += model_param.cw**mdc[0] * ( state.pw_mdc[mdc] - psc )

        # Cost of Cancelling
        for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
            if ttpmdc[2] == 0: continue
            if ttpmdc[0] > ttpmdc[1]: #good schedule
                cost -= model_param.cc * action.rsc_ttpmdc[ttpmdc]
            elif ttpmdc[1] > ttpmdc[0]: #bad schedule
                cost += model_param.cc * action.rsc_ttpmdc[ttpmdc]

        # Violating unit bounds
        for tp in itertools.product(indices['t'], indices['p']):
            cost += state.uv_tp[tp] * M

        return(cost)

    # Generates beta 0 constraint data 
    def b_0_constraint(state: state, action: action) -> constraint_parameter:

        constr_data = constraint_parameter(
            {"b_0": 1-gamma}, 
            {"b_0": 1},
            {"b_0": '='},
            {"b_0": 'beta_0'},
        )
        return constr_data

    # Generates beta ue constraint data
    def b_ue_constraint(state: state, action: action) -> constraint_parameter:
        # Initializes Data
        lhs = {}
        rhs = {}
        sign = {}
        name = {}

        # Creates Data
        for tp in itertools.product(indices['t'], indices['p']):
            lhs[tp] = state.ue_tp[tp] - gamma*(ppe_data[tp[1]].expected_units)
            rhs[tp] = input_data.expected_state_values['ue'][tp]
            sign[tp] = ">="
            name[tp] = f"beta_ue_tp_{tp[0]}_{tp[1].replace(' ', '-')}"

        # Returns Data
        constr_data = constraint_parameter(lhs,rhs,sign,name)
        return constr_data

    # Generates beta uu constraint data
    def b_uu_constraint(state: state, action: action) -> constraint_parameter:
        # Initializes Data
        lhs = {}
        rhs = {}
        sign = {}
        name = {}

        # Creates Data
        for tp in itertools.product(indices['t'], indices['p']):
            if tp[0] == indices['t'][-1]:
                lhs[tp] = state.uu_tp[tp]
            else:
                parameter_val = state.uu_tp[tp]
                parameter_val -= gamma * (state.uu_tp[(tp[0] + 1, tp[1])])

                # Calculates impact of transition in difficulties
                for mc in itertools.product(indices['m'],indices['c']):
                    for d in range(len(indices['d'])):
                        if mc[0] == 0: continue
                        
                        transition_prob = transition[(mc[0], indices['d'][d], mc[1])]
                        ps_val = state.ps_tmdc[(tp[0], mc[0], indices['d'][d], mc[1])]
                        if d == len(indices['d'])-1: usage_change = 0
                        else: usage_change = usage[(tp[1], indices['d'][d+1], mc[1])] - usage[(tp[1], indices['d'][d], mc[1])]

                        parameter_val -= gamma * ps_val * transition_prob * usage_change

                # Calculates impact of scheduling
                for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
                    parameter_val -= action.sc_tmdc[(tp[0], mdc[0], mdc[1], mdc[2])] * usage[(tp[1], mdc[1], mdc[2])]

                # Calcualtes impact of rescheduling into it
                for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
                    if tmdc[1] == 0: continue
                    parameter_val -= action.rsc_ttpmdc[(tmdc[0], tp[0]+1, tmdc[1], tmdc[2], tmdc[3])] * usage[(tp[1], tmdc[2], tmdc[3])]

                # Calcualtes impact of rescheduling out of it
                for tpmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
                    if tpmdc[1] == 0: continue
                    parameter_val += action.rsc_ttpmdc[(tp[0]+1, tpmdc[0], tpmdc[1], tpmdc[2], tpmdc[3])] * usage[(tp[1], tpmdc[2], tpmdc[3])]
                
                lhs[tp] = parameter_val
            
            rhs[tp] = input_data.expected_state_values['uu'][tp]
            sign[tp] = ">="
            name[tp] = f"beta_uu_tp_{tp[0]}_{tp[1].replace(' ', '-')}"
        
        # Returns Data
        constr_data = constraint_parameter(lhs,rhs,sign,name)
        return constr_data

    # Generates beta uv constraint data
    def b_uv_constraint(state: state, action: action) -> constraint_parameter:
        # Initializes Data
        lhs = {}
        rhs = {}
        sign = {}
        name = {}

        # Creates Data
        for tp in itertools.product(indices['t'], indices['p']):
            lhs[tp] = state.uv_tp[tp]
            rhs[tp] = input_data.expected_state_values['uv'][tp]
            sign[tp] = ">="
            name[tp] = f"beta_uv_tp_{tp[0]}_{tp[1].replace(' ', '-')}"

        # Returns Data
        constr_data = constraint_parameter(lhs,rhs,sign,name)
        return constr_data

    # Generates beta pe constraint data
    def b_pe_constraint(state: state, action: action) -> constraint_parameter:
        # Initializes Data
        lhs = {}
        rhs = {}
        sign = {}
        name = {}
        
        # Creates Data
        for dc in itertools.product(indices['d'], indices['c']):
            lhs[dc] = state.pe_dc[dc] - gamma*(arrival[dc])
            rhs[dc] = input_data.expected_state_values['pe'][dc]
            sign[dc] = ">="
            name[dc] = f"beta_pe_dc_{dc[0].replace(' ','-')}_{dc[1].replace(' ', '-')}"
        
        # Returns Data
        constr_data = constraint_parameter(lhs,rhs,sign,name)
        return constr_data

    # Generates beta pw constraint data
    def b_pw_constraint(state: state, action: action) -> constraint_parameter:
        # Initializes Data
        lhs = {}
        rhs = {}
        sign = {}
        name = {}
        
        # Creates Data
        for mc in itertools.product(indices['m'], indices['c']):
            for d in range(len(indices['d'])):
                mdc = (mc[0], indices['d'][d], mc[1])

                if mc[0] == 0: continue
                
                # When m = 1
                elif mc[0] == 1:
                    curr_state = state.pw_mdc[mdc]
                    not_scheduled = gamma * state.pe_dc[(indices['d'][d],mc[1])]
                    for t in indices['t']:
                        not_scheduled -= gamma * action.sc_tmdc[(t, mc[0]-1, indices['d'][d], mc[1])]
                    lhs[mdc] = curr_state - not_scheduled

                # When m = M
                elif mc[0] == indices['m'][-1]:
                    curr_state = state.pw_mdc[mdc]
                    not_scheduled = 0
                    transitioned_out = 0
                    transitioned_in = 0

                    for mm in input_data.indices['m'][-2:]:
                        not_scheduled += gamma * state.pw_mdc[(mm, indices['d'][d], mc[1])]
                        for t in indices['t']:
                            not_scheduled -= gamma * action.sc_tmdc[(t, mm, indices['d'][d], mc[1])]
                    transitioned_out = gamma * state.pw_mdc[(mm,indices['d'][d],mc[1])] * transition[(mm,indices['d'][d],mc[1])]
                    if d == 0: 
                        transitioned_in = 0
                    else:
                        transitioned_in = gamma * state.pw_mdc[(mm,indices['d'][d-1],mc[1])] * transition[(mm,indices['d'][d-1],mc[1])]
                        
                    lhs[mdc] = curr_state - not_scheduled + transitioned_out - transitioned_in

                # All others
                else:                   
                    curr_state = state.pw_mdc[mdc]
                    not_scheduled = gamma * state.pw_mdc[(mc[0]-1, indices['d'][d], mc[1])]
                    for t in indices['t']:
                        not_scheduled -= gamma * action.sc_tmdc[(t, mc[0]-1, indices['d'][d], mc[1])]
                    transitioned_out = gamma * state.pw_mdc[(mc[0]-1,indices['d'][d],mc[1])] * transition[(mc[0]-1,indices['d'][d],mc[1])]
                    if d == 0: 
                        transitioned_in = 0
                    else:
                        transitioned_in = gamma * state.pw_mdc[(mc[0]-1,indices['d'][d-1],mc[1])] * transition[(mc[0]-1,indices['d'][d-1],mc[1])]
                    lhs[mdc] = curr_state - not_scheduled + transitioned_out - transitioned_in

                rhs[mdc] = input_data.expected_state_values['pw'][mdc]
                sign[mdc] = ">="
                name[mdc] = f"beta_pw_mdc_{mdc[0]}_{mdc[1].replace(' ', '-')}_{mdc[2].replace(' ', '-')}"
            
            # Returns Data
        constr_data = constraint_parameter(lhs,rhs,sign,name)
        return constr_data

    # Generates beta ps constraint data
    def b_ps_constraint(state: state, action: action) -> constraint_parameter:
        pass

    # generates constraints for a model
    def generate_constraint( 
        model: gp.Model,
        state_action_variable: List[gp.Var],
        state_action_data: List[ Tuple[state,action] ], 
        generator_function: Callable[ [state,action], constraint_parameter ],
    ) -> Dict[ Tuple[str],gp.Constr ]:
        
        # Initialization
        expr: Dict[ Tuple[str],gp.LinExpr ] = {}
        init_iterables = generator_function(state_action_data[0][0], state_action_data[0][1])
        for index_iter in init_iterables.lhs_param.keys():
            expr[index_iter] = gp.LinExpr()
        
        # Generation
        for sa_pair in range(len(state_action_data)):
            params = generator_function(state_action_data[sa_pair][0], state_action_data[sa_pair][1])
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
    
    # Model
    mast_model = gp.Model('MasterKnapsack')

    # Generates Variables
    w_sa_var = []
    for sa_pair in range(len(state_action_data)):
        w_sa_var.append(mast_model.addVar(name=f'sa_{sa_pair}'))

    # Generates Constraints
    b0_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_0_constraint)
    ue_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_ue_constraint)
    uu_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_uu_constraint)
    uv_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_uv_constraint)
    pe_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_pe_constraint)
    pw_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_pw_constraint)
    # ps_constr = generate_constraint(mast_model, w_sa_var, state_action_data, b_ps_constraint)
    constraints = {
        'b0': b0_constr,
        'ue': ue_constr,
        'uu': uu_constr,
        'uv': uv_constr,
        'pe': pe_constr,
        'pw': pw_constr
    }

    # Generates Objective Function
    obj_expr = gp.LinExpr()
    for sa_pair in range(len(state_action_data)):
        cost_param = cost_function(state_action_data[sa_pair][0], state_action_data[sa_pair][1])
        obj_expr.add(w_sa_var[sa_pair], cost_param)
    mast_model.setObjective(obj_expr, GRB.MINIMIZE)

    # Returns model
    return mast_model, w_sa_var, constraints
# %%
