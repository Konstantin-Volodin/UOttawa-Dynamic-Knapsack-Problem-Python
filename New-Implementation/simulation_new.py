#%%
##### Initialization & Changeable Parameters #####
#region
from lib2to3.pgen2.token import NEWLINE
from os import linesep
import csv

from numpy.core.fromnumeric import prod
from Modules import data_import

from gurobipy import *
import itertools
import os.path
import pickle
from copy import deepcopy
import numpy as np
from tqdm.auto import trange
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

#endregion

def main_func(replications, warm_up, duration, show_policy, import_data, import_beta, export_txt, export_pic, export_state_my, export_state_md, export_cost_my, export_cost_md, export_util_my, export_util_md, export_sa_my, export_sa_md):
    ##### Read Data #####
    #region

    my_path = os.getcwd()
    input_data = data_import.read_data(os.path.join(my_path, import_data))
    
    export_state_my = os.path.join(my_path,export_state_my)
    export_state_md = os.path.join(my_path,export_state_md)

    # Quick Assess to Various Parameters
    TL = input_data.transition.wait_limit
    BM = 10000
    max_lims = 10000
    U = input_data.usage
    p_dat = input_data.ppe_data
    pea = input_data.arrival
    gam = input_data.model_param.gamma
    ptp_d = input_data.transition.transition_rate_comp
    ptp_k = input_data.transition.transition_rate_pri
    cw = input_data.model_param.cw
    cs = input_data.model_param.cs
    cc = input_data.model_param.cc
    cv = input_data.model_param.cv

    ##### Generating Sets #####
    T = input_data.indices['t']
    M = input_data.indices['m']
    P = input_data.indices['p']
    D = input_data.indices['d']
    K = input_data.indices['k']
    C = input_data.indices['c']

    # Sub Sets
    PCO = []
    PNCO = []
    for k, v in input_data.ppe_data.items(): 
        if v.ppe_type == "non-carry-over": PNCO.append(k) 
        else: PCO.append(k)
        
    mTLdkc = [(m, d, k, c) for c in C for m in M[1:input_data.transition.wait_limit[c]] for d in D for k in K ]
    TLMdkc = [(m, d, k, c) for c in C for m in M[input_data.transition.wait_limit[c]:-1] for d in D for k in K ]
    tmTLdkc = [(t, m, d, k, c) for t in T[:-1] for c in C for m in M[1:input_data.transition.wait_limit[c]] for d in D for k in K ]
    tTLMdkc = [(t, m, d, k, c) for t in T[:-1] for c in C for m in M[input_data.transition.wait_limit[c]:-1] for d in D for k in K ]

    # Expected Data 
    E_UL = input_data.expected_state_values['ul']
    E_PW = input_data.expected_state_values['pw']
    E_PS = input_data.expected_state_values['ps']

    # Initial State
    init_state = {'ul': E_UL, 'pw': E_PW, 'ps': E_PS}
    for i in itertools.product(M,D,K,C): init_state['pw'][i] = 0
    for i in itertools.product(T,M,D,K,C): init_state['ps'][i] = 0
    for i in itertools.product(P): init_state['ul'][i] = 0

    # Review Importance
    # for m in M:
    #     init_state['pw'][(m, 'Complexity 1', 'P2', '6. SPINE POSTERIOR DISCECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 1', 'P3', '6. SPINE POSTERIOR DISCECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P2', '6. SPINE POSTERIOR DISCECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 1', 'P2', '1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 1', 'P3', '1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 1', 'P2', '4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P3', '6. SPINE POSTERIOR DISCECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P2', '4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P3', '4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P2', '1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR')] = 0
    #     init_state['pw'][(m, 'Complexity 1', 'P3', '4. SPINE POST CERV DECOMPRESSION AND FUSION W INSTR')] = 0
    #     init_state['pw'][(m, 'Complexity 2', 'P3', '1. SPINE POSTERIOR DECOMPRESSION/LAMINECTOMY LUMBAR')] = 0

    for k,v in init_state['pw'].items():
        print(k,v)

    # Betas
    with open(os.path.join(my_path, import_beta), 'rb') as handle:
        betas = pickle.load(handle)
    #endregion

    ##### Various Functions #####
    #region
    def non_zero_state(state):
        ret_str = ""
        non_zero_st = {'ul': {}, 'pw': {}, 'ps': {}}
        for k,v in state['ul'].items(): 
            if v != 0: 
                ret_str += f'\tState - Units Left Over - {k} - {v}\n'
                # print(f'\tState - Units Left Over - {k} - {v}')
                # non_zero_st['ul'][k] = v
        for k,v in state['pw'].items(): 
            if v != 0: 
                ret_str += f'\tState - Patients Waiting - {k} - {v}\n'
                # print(f'\tState - Patients Waiting - {k} - {v}')
                # non_zero_st['pw'][k] = v
        for k,v in state['ps'].items(): 
            if v != 0: 
                ret_str += f'\tState - Patients Scheduled - {k} - {v}\n'
                # print(f'\tState - Patients Scheduled - {k} - {v}')
                # non_zero_st['ps'][k] = v
        return(ret_str)

    def non_zero_action(action):
        ret_str = ""
        non_zero_ac = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
        for k,v in action['sc'].items():
            if v != 0: 
                ret_str += f'\tAction - Schedule Patients - {k} - {v}\n'
                # print(f'\tAction - Schedule Patients - {k} - {v}')
                # non_zero_ac['sc'][k] = v
        for k,v in action['rsc'].items():
            if v != 0: 
                ret_str += f'\tAction - Reschedule Patients - {k} - {v}\n'
                # print(f'\tAction - Reschedule Patients - {k} - {v}')
                # non_zero_ac['rsc'][k] = v
        for k,v in action['uv'].items():
            if v != 0: 
                ret_str += f'\tAction - Units Violated - {k} - {v}\n'
                # print(f'\tAction - Units Violated - {k} - {v}')
                # non_zero_ac['uv'][k] = v
        for k,v in action['ulp'].items():
            if v != 0: 
                ret_str += f'\tPost Action - Units Left Over - {k} - {v}\n'
                # print(f'\tPost Action - Units Left Over - {k} - {v}')
                # non_zero_ac['ulp'][k] = v
        for k,v in action['pwp'].items():
            if v != 0: 
                ret_str += f'\tPost Action - Patients Waiting - {k} - {v}\n'
                # print(f'\tPost Action - Patients Waiting - {k} - {v}')
                # non_zero_ac['pwp'][k] = v
        for k,v in action['psp'].items():
            if v != 0: 
                ret_str += f'\tPost Action - Patients Scheduled - {k} - {v}\n'
                # print(f'\tPost Action - Patients Scheduled - {k} - {v}')
                # non_zero_ac['psp'][k] = v
        for k,v in action['uup'].items():
            if v != 0: 
                ret_str += f'\tPost Action - Units Used - {k} - {v}\n'
                # print(f'\tPost Action - Units Used - {k} - {v}')
                # non_zero_ac['uup'][k] = v
        return(ret_str)
    
    def retrive_surg_subset(df, d, k, c, m, day):
        '''
        # Given a patient dataset, retrieves patient who are specific surgery type (accomodates transitions)
        # Parameters: df - dataset, d - complexity, k - priority, c - surgery type, m - filter by wait time, day - current day
        '''
        # Filter on complexity, priority, and surgery type
        df_last_rows = df.groupby('id').tail(1).reset_index()
        df_surg_subset = df_last_rows.query(f"priority=='{k}' and complexity=='{d}' and surgery=='{c}'")
        df_surg_full_subset = df[df['id'].isin(df_surg_subset['id'])]

        # Filter on wait time
        if m != M[-1]:
            df_wait_subset = df_surg_full_subset.query(f"arrived_on == {day-m}")
            df_wat_full_subset = df[df['id'].isin(df_wait_subset['id'])]
        else:
            df_wait_subset = df_surg_full_subset.query(f"action == 'arrived'").query(f"arrived_on <= {day-m}")
            df_wat_full_subset = df[df['id'].isin(df_wait_subset['id'])]

        return(df_wat_full_subset)
        
    #endregion

    ##### Myopic Model #####
    #region
    myopic = Model('Myopic')
    myopic.params.LogToConsole = 0

    # Cost Params
    myv_cost_cw = myopic.addVars(K, vtype=GRB.CONTINUOUS, name='var_cost_cw')
    myv_cost_cs = myopic.addVars(K,[0]+T, vtype=GRB.CONTINUOUS, name='var_cost_cs')
    myv_cost_cc = myopic.addVars(K, vtype=GRB.CONTINUOUS, name='var_cost_cc')
    myv_cost_cv = myopic.addVar(vtype=GRB.CONTINUOUS, name='var_cost_cv')

    # Fix Costs
    for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
    for t,k in itertools.product(T, K): myv_cost_cs[(k,t)].UB = cs[k][t]; myv_cost_cs[(k,t)].LB = cs[k][t];
    for k in K: myv_cost_cc[k].UB = cc[k]; myv_cost_cc[k].LB = cc[k];
    for k in K: myv_cost_cv.UB = cv; myv_cost_cv.LB = cv;


    # State Action & Auxiliary Variables
    myv_st_ul = myopic.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, name='var_state_ul')
    myv_st_pw = myopic.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_pw')
    myv_st_ps = myopic.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_ps')

    myv_ac_sc = myopic.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
    myv_ac_rsc = myopic.addVars(T, T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_rsc')

    myv_aux_uv = myopic.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
    myv_aux_uvb = myopic.addVars(T, P, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

    myv_aux_ulp = myopic.addVars(PCO, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
    myv_aux_ulb = myopic.addVars(PCO, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

    myv_aux_uup = myopic.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
    myv_aux_pwp = myopic.addVars(M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
    myv_aux_psp = myopic.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

    myv_aux_pwt_d = myopic.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
    myv_aux_pwt_k = myopic.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
    myv_aux_pst_d = myopic.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
    myv_aux_pst_k = myopic.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

    # Definition of auxiliary variables
    myc_uup = myopic.addConstrs((myv_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * myv_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
    myc_pwp = myopic.addConstrs((myv_aux_pwp[(m,d,k,c)] == myv_st_pw[(m,d,k,c)] - quicksum( myv_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
    myc_psp = myopic.addConstrs((myv_aux_psp[(t,m,d,k,c)] == myv_st_ps[(t,m,d,k,c)] + myv_ac_sc[(t,m,d,k,c)] + quicksum( myv_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( myv_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

    myc_aux_uv_0 = myopic.addConstrs((myv_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
    myc_aux_uv_0M = myopic.addConstrs((myv_aux_uv[(t,p)] <= BM * myv_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
    myc_aux_uv_1 = myopic.addConstrs((myv_aux_uv[(1,p)] >= myv_aux_uup[(1, p)] - p_dat[p].expected_units - myv_st_ul[p] for p in P), name='con_auxiliary_uv_1')
    myc_aux_uv_1M = myopic.addConstrs((myv_aux_uv[(1,p)] <= (myv_aux_uup[(1, p)] - p_dat[p].expected_units - myv_st_ul[p]) + BM * (1 - myv_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
    myc_aux_uv_m = myopic.addConstrs((myv_aux_uv[(t, p)] >= (myv_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
    myc_aux_uv_mM = myopic.addConstrs((myv_aux_uv[(t, p)] <= (myv_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - myv_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

    myc_aux_ulp_0 = myopic.addConstrs((myv_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
    myc_aux_ulp_0M = myopic.addConstrs((myv_aux_ulp[p] <= BM * myv_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
    myc_aux_ulp_p = myopic.addConstrs((myv_aux_ulp[p] >= (p_dat[p].expected_units + myv_st_ul[p] - myv_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
    myc_aux_ulp_pM = myopic.addConstrs((myv_aux_ulp[p] <= (p_dat[p].expected_units + myv_st_ul[p] - myv_aux_uup[(1,p)]) + BM * (1-myv_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

    myc_aux_pwt_d = myopic.addConstrs((myv_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * myv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_d')
    myc_aux_pwt_k_0 = myopic.addConstrs((myv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_pwp[(m,d,k,c)] - myv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pwt_k_0')
    myc_aux_pwt_k_i = myopic.addConstrs((myv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_pwp[(m,d,k,c)] + myv_aux_pwt_d[(m,D[D.index(d)-1],k,c)] - myv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pwt_k')
    myc_aux_pwt_k_D = myopic.addConstrs((myv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_pwp[(m,d,k,c)] + myv_aux_pwt_d[(m,D[D.index(d)-1],k,c)]) for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pwt_k_D')
    myc_aux_pwt_k = {**myc_aux_pwt_k_0, **myc_aux_pwt_k_i, **myc_aux_pwt_k_D}


    myc_aux_pst_d = myopic.addConstrs((myv_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * myv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pst_d')
    myc_aux_pst_k_0 = myopic.addConstrs((myv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_psp[(t,m,d,k,c)] - myv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pst_k_0')
    myc_aux_pst_k_i = myopic.addConstrs((myv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_psp[(t,m,d,k,c)] + myv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)] - myv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pst_k')
    myc_aux_pst_k_D = myopic.addConstrs((myv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (myv_aux_psp[(t,m,d,k,c)] + myv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pst_k_D')
    myc_aux_pst_k = {**myc_aux_pst_k_0, **myc_aux_pst_k_i, **myc_aux_pst_k_D}

    # State Action Constraints
    myc_usage_1 = myopic.addConstrs((myv_aux_uup[(1,p)] <= p_dat[p].expected_units + myv_st_ul[p] + myv_aux_uv[(1,p)] for p in P), name='con_usage_1')
    myc_usage_tT = myopic.addConstrs((myv_aux_uup[(t,p)] <= p_dat[p].expected_units + myv_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

    myc_rescb = myopic.addConstrs((myv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
    myc_rescb_ttp = myopic.addConstrs((myv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

    myc_resch = myopic.addConstrs((quicksum(myv_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= myv_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
    myc_sched = myopic.addConstrs((quicksum(myv_ac_sc[(t,m,d,k,c)] for t in T) <= myv_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

    # Objective Value
    myo_cw =     quicksum( myv_cost_cw[k] * myv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    myo_cs =     quicksum( myv_cost_cs[(k,t)] * myv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    myo_brsc =   quicksum( (myv_cost_cs[(k,tp-t)]+myv_cost_cc[k]) * myv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    myo_grsc =   quicksum( (myv_cost_cs[(k,t-tp)]-myv_cost_cc[k]) * myv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    myo_cv =     quicksum( myv_cost_cv * myv_aux_uv[(t,p)] for  t in T for p in P)
    myo_cost = (myo_cw + myo_cs + myo_brsc - myo_grsc + myo_cv)
    myopic.setObjective( myo_cost, GRB.MINIMIZE )
    #endregion

    ##### MDP Model #####
    #region
    MDP = Model('MDP')
    MDP.params.LogToConsole = 0

    # State Action & Auxiliary Variables
    mdv_st_ul = MDP.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, name='var_state_ul')
    mdv_st_pw = MDP.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_pw')
    mdv_st_ps = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_ps')

    mdv_ac_sc = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
    mdv_ac_rsc = MDP.addVars(T, T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_rsc')

    mdv_aux_uv = MDP.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
    mdv_aux_uvb = MDP.addVars(T, P, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

    mdv_aux_ulp = MDP.addVars(PCO, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
    mdv_aux_ulb = MDP.addVars(PCO, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

    mdv_aux_uup = MDP.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
    mdv_aux_pwp = MDP.addVars(M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
    mdv_aux_psp = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

    mdv_aux_pwt_d = MDP.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
    mdv_aux_pwt_k = MDP.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
    mdv_aux_pst_d = MDP.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
    mdv_aux_pst_k = MDP.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

    # Definition of auxiliary variables
    mdc_uup = MDP.addConstrs((mdv_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * mdv_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
    mdc_pwp = MDP.addConstrs((mdv_aux_pwp[(m,d,k,c)] == mdv_st_pw[(m,d,k,c)] - quicksum( mdv_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
    mdc_psp = MDP.addConstrs((mdv_aux_psp[(t,m,d,k,c)] == mdv_st_ps[(t,m,d,k,c)] + mdv_ac_sc[(t,m,d,k,c)] + quicksum( mdv_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( mdv_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

    mdc_aux_uv_0 = MDP.addConstrs((mdv_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
    mdc_aux_uv_0M = MDP.addConstrs((mdv_aux_uv[(t,p)] <= BM * mdv_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
    mdc_aux_uv_1 = MDP.addConstrs((mdv_aux_uv[(1,p)] >= mdv_aux_uup[(1, p)] - p_dat[p].expected_units - mdv_st_ul[p] for p in P), name='con_auxiliary_uv_1')
    mdc_aux_uv_1M = MDP.addConstrs((mdv_aux_uv[(1,p)] <= (mdv_aux_uup[(1, p)] - p_dat[p].expected_units - mdv_st_ul[p]) + BM * (1 - mdv_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
    mdc_aux_uv_m = MDP.addConstrs((mdv_aux_uv[(t, p)] >= (mdv_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
    mdc_aux_uv_mM = MDP.addConstrs((mdv_aux_uv[(t, p)] <= (mdv_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - mdv_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

    mdc_aux_ulp_0 = MDP.addConstrs((mdv_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
    mdc_aux_ulp_0M = MDP.addConstrs((mdv_aux_ulp[p] <= BM * mdv_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
    mdc_aux_ulp_p = MDP.addConstrs((mdv_aux_ulp[p] >= (p_dat[p].expected_units + mdv_st_ul[p] - mdv_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
    mdc_aux_ulp_pM = MDP.addConstrs((mdv_aux_ulp[p] <= (p_dat[p].expected_units + mdv_st_ul[p] - mdv_aux_uup[(1,p)]) + BM * (1-mdv_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

    mdc_aux_pwt_d = MDP.addConstrs((mdv_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * mdv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwt_d')
    mdc_aux_pwt_k_0 = MDP.addConstrs((mdv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_pwp[(m,d,k,c)] - mdv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pwt_k_0')
    mdc_aux_pwt_k_i = MDP.addConstrs((mdv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_pwp[(m,d,k,c)] + mdv_aux_pwt_d[(m,D[D.index(d)-1],k,c)] - mdv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pwt_k')
    mdc_aux_pwt_k_D = MDP.addConstrs((mdv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_pwp[(m,d,k,c)] + mdv_aux_pwt_d[(m,D[D.index(d)-1],k,c)]) for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pwt_k_D')
    mdc_aux_pwt_k = {**mdc_aux_pwt_k_0, **mdc_aux_pwt_k_i, **mdc_aux_pwt_k_D}

    mdc_aux_pst_d = MDP.addConstrs((mdv_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * mdv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pst_d')
    mdc_aux_pst_k_0 = MDP.addConstrs((mdv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_psp[(t,m,d,k,c)] - mdv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pst_k_0')
    mdc_aux_pst_k_i = MDP.addConstrs((mdv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_psp[(t,m,d,k,c)] + mdv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)] - mdv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pst_k')
    mdc_aux_pst_k_D = MDP.addConstrs((mdv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (mdv_aux_psp[(t,m,d,k,c)] + mdv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pst_k_D')
    mdc_aux_pst_k = {**mdc_aux_pst_k_0, **mdc_aux_pst_k_i, **mdc_aux_pst_k_D}


    # State Action Constraints
    mdc_usage_1 = MDP.addConstrs((mdv_aux_uup[(1,p)] <= p_dat[p].expected_units + mdv_st_ul[p] + mdv_aux_uv[(1,p)] for p in P), name='con_usage_1')
    mdc_usage_tT = MDP.addConstrs((mdv_aux_uup[(t,p)] <= p_dat[p].expected_units + mdv_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

    mdc_rescb = MDP.addConstrs((mdv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
    mdc_rescb_ttp = MDP.addConstrs((mdv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

    mdc_resch = MDP.addConstrs((quicksum(mdv_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= mdv_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
    mdc_sched = MDP.addConstrs((quicksum(mdv_ac_sc[(t,m,d,k,c)] for t in T) <= mdv_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

    # Objective Value
    mdo_val_ul_co = quicksum( betas['bul'][p] * mdv_aux_ulp[p] for p in PCO )

    mdo_val_pw_0 = quicksum( betas['bpw'][(0,d,k,c)] * pea[(d,k,c)] for d in D for k in K for c in C )
    mdo_val_pw_1TL = quicksum( betas['bpw'][(m,d,k,c)] * mdv_aux_pwp[(m-1,d,k,c)] for m,d,k,c in mTLdkc )
    mdo_val_pw_TLM = (
        quicksum( betas['bpw'][(m,d,k,c)] * mdv_aux_pwp[(m-1,d,k,c)] for m,d,k,c in TLMdkc ) + 
        quicksum( betas['bpw'][(m,d,k,c)] * mdv_aux_pwt_d[(m-1, D[D.index(d)-1] ,k,c)] for m,d,k,c in TLMdkc if d != D[0] ) +
        quicksum( betas['bpw'][(m,d,k,c)] * mdv_aux_pwt_k[(m-1,d, K[K.index(k)-1] ,c)] for m,d,k,c in TLMdkc if k != K[0] ) +
        quicksum( -betas['bpw'][(m,d,k,c)] * mdv_aux_pwt_d[(m-1,d,k,c)] for m,d,k,c in TLMdkc if d != D[-1] ) +
        quicksum( -betas['bpw'][(m,d,k,c)] * mdv_aux_pwt_k[(m-1,d,k,c)] for m,d,k,c in TLMdkc if k != K[-1] )
    )
    mdo_val_pw_M = (
        quicksum( betas['bpw'][(M[-1],d,k,c)] * mdv_aux_pwp[(mm,d,k,c)] for mm in M[-2:] for d in D for k in K for c in C ) + 
        quicksum( betas['bpw'][(M[-1],d,k,c)] * mdv_aux_pwt_d[(mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:] for d in D for k in K for c in C if d != D[0] ) +
        quicksum( betas['bpw'][(M[-1],d,k,c)] * mdv_aux_pwt_k[(mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:] for d in D for k in K for c in C if k != K[0] ) +
        quicksum( -betas['bpw'][(M[-1],d,k,c)] * mdv_aux_pwt_d[(mm,d,k,c)] for mm in M[-2:] for d in D for k in K for c in C if d != D[-1] ) +
        quicksum( -betas['bpw'][(M[-1],d,k,c)] * mdv_aux_pwt_k[(mm,d,k,c)] for mm in M[-2:] for d in D for k in K for c in C if k != K[-1] )
    )

    mdo_val_ps_1TL = quicksum( betas['bps'][(t,m,d,k,c)] * mdv_aux_psp[(t+1,m-1,d,k,c)] for t,m,d,k,c in tmTLdkc )
    mdo_val_ps_TLM = (
        quicksum( betas['bps'][(t,m,d,k,c)] * mdv_aux_psp[(t+1,m-1,d,k,c)] for t,m,d,k,c in tTLMdkc ) + 
        quicksum( betas['bps'][(t,m,d,k,c)] * mdv_aux_pst_d[(t+1,m-1, D[D.index(d)-1] ,k,c)] for t,m,d,k,c in tTLMdkc if d != D[0] ) +
        quicksum( betas['bps'][(t,m,d,k,c)] * mdv_aux_pst_k[(t+1,m-1,d, K[K.index(k)-1] ,c)] for t,m,d,k,c in tTLMdkc if k != K[0] ) +
        quicksum( -betas['bps'][(t,m,d,k,c)] * mdv_aux_pst_d[(t+1,m-1,d,k,c)] for t,m,d,k,c in tTLMdkc if d != D[-1] ) +
        quicksum( -betas['bps'][(t,m,d,k,c)] * mdv_aux_pst_k[(t+1,m-1,d,k,c)] for t,m,d,k,c in tTLMdkc if k != K[-1] )
    )
    mdo_val_ps_M = (
        quicksum( betas['bps'][(t,M[-1],d,k,c)] * mdv_aux_psp[(t+1,mm,d,k,c)] for mm in M[-2:] for t in T[:-1] for d in D for k in K for c in C ) + 
        quicksum( betas['bps'][(t,M[-1],d,k,c)] * mdv_aux_pst_d[(t+1,mm, D[D.index(d)-1] ,k,c)] for t in T[:-1] for mm in M[-2:] for d in D for k in K for c in C if d != D[0] ) +
        quicksum( betas['bps'][(t,M[-1],d,k,c)] * mdv_aux_pst_k[(t+1,mm,d, K[K.index(k)-1] ,c)] for t in T[:-1] for mm in M[-2:] for d in D for k in K for c in C if k != K[0] ) +
        quicksum( -betas['bps'][(t,M[-1],d,k,c)] * mdv_aux_pst_d[(t+1,mm,d,k,c)] for t in T[:-1] for mm in M[-2:] for d in D for k in K for c in C if d != D[-1] ) +
        quicksum( -betas['bps'][(t,M[-1],d,k,c)] * mdv_aux_pst_k[(t+1,mm,d,k,c)] for t in T[:-1] for mm in M[-2:] for d in D for k in K for c in C if k != K[-1] )
    )
    mdo_val = (betas['b0'] + (mdo_val_ul_co) + (mdo_val_pw_0+mdo_val_pw_1TL+mdo_val_pw_TLM+mdo_val_pw_M) + (mdo_val_ps_1TL+mdo_val_ps_TLM+mdo_val_ps_M))

    mdo_cw =     quicksum( cw[k] * mdv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    mdo_cs =     quicksum( cs[k][t] * mdv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    mdo_brsc =   quicksum( (cs[k][tp-t]+cc[k]) * mdv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    mdo_grsc =   quicksum( (cs[k][t-tp]-cc[k]) * mdv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    mdo_cv =     quicksum( cv * mdv_aux_uv[(t,p)] for  t in T for p in P)
    mdo_cost = (mdo_cw + mdo_cs + mdo_brsc - mdo_grsc + mdo_cv)

    MDP.setObjective( mdo_cost+(gam*mdo_val), GRB.MINIMIZE )
    #endregion
    
    ##### Myopic Simulation
    #region

    # Simulation Data
    my_sim_st = []
    my_sim_ac = []
    my_sim_cost = []
    my_sim_disc = []

    text_file = open(export_state_my, 'w', newline="")
    logging_file = open(export_sa_my, 'w', newline="")
    cost_file = open(export_cost_my, 'w', newline="")
    util_file = open(export_util_my, 'w', newline="")
    print(f"repl,period,cost,cost_cw,cost_cs,cost_brsc,cost_grsc,cost_cv", file=cost_file)
    print(f"repl,period,horizon_period,usage_admin,usage_OR", file=util_file)
    print(f"repl,period,state-aciton,value,t,tp,m,d,k,c,p,val", file=logging_file)

    # Simulation
    for repl in trange(replications, desc='Myopic'):

        # Random streams
        my_strm_dev = np.random.default_rng(repl)
        my_strm_pwt = np.random.default_rng(repl)
        my_strm_pst = np.random.default_rng(repl)
        my_strm_arr = np.random.default_rng(repl)
            
        # Aggregate Replication Data
        rp_st = []
        rp_ac = []
        rp_cost = []
        rp_disc = 0
        state = deepcopy(init_state)

        # Detailed Logging Data
        patient_data_df = pd.DataFrame( columns=['repl', 'period', 'policy', 'id','priority', 'complexity', 'surgery','action','arrived_on','sched_to', 'resch_from', 'resch_to', 'transition'] )
        patient_id_count = 0

        # Single Replication
        for day in trange(duration):

            # Init Patient Data
            patient_data = {'repl': [], 'period': [], 'policy': [], 'id': [], 'priority': [], 'complexity': [], 'surgery': [], 'action': [], 'arrived_on': [], 'sched_to': [], 'resch_from': [], 'resch_to': [], 'transition': []}
            if day == 0:
                for m,d,k,c in itertools.product(M,D,K,C):
                    for elem in range(int(state['pw'][(m,d,k,c)])):  
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(patient_id_count)
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('arrived')
                        patient_data['arrived_on'].append(day-m)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append(pd.NA)
                        patient_id_count += 1
                patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                patient_data = {k : [] for k in patient_data}


            # Generate Action (With slightly different logic)
            for k in K: myv_cost_cw[k].UB = cv-1; myv_cost_cw[k].LB = cv-1;
            for p in P: myv_st_ul[p].UB = state['ul'][p]; myv_st_ul[p].LB = state['ul'][p]
            for m,d,k,c in itertools.product(M, D, K, C): myv_st_pw[(m,d,k,c)].UB = round(state['pw'][(m,d,k,c)],0); myv_st_pw[(m,d,k,c)].LB = round(state['pw'][(m,d,k,c)],0)
            for t,m,d,k,c in itertools.product(T, M, D, K, C): myv_st_ps[(t,m,d,k,c)].UB = round(state['ps'][(t,m,d,k,c)],0); myv_st_ps[(t,m,d,k,c)].LB = round(state['ps'][(t,m,d,k,c)],0)
            myopic.optimize()

            # Save Cost (with normal logic)
            for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
            for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = round(myv_ac_sc[(t,m,d,k,c)].X,0); myv_ac_sc[(t,m,d,k,c)].LB = round(myv_ac_sc[(t,m,d,k,c)].X,0)
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = round(myv_ac_rsc[(t,tp,m,d,k,c)].X,0); myv_ac_rsc[(t,tp,m,d,k,c)].LB = round(myv_ac_rsc[(t,tp,m,d,k,c)].X,0)
            myopic.optimize()

            # Reset Myopic
            for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = GRB.INFINITY; myv_ac_sc[(t,m,d,k,c)].LB = 0
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = GRB.INFINITY; myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0

            rp_cost.append(myo_cost.getValue())
            print(f"{repl},{day}, {myo_cost.getValue()},{myo_cw.getValue()},{myo_cs.getValue()},{myo_brsc.getValue()},{myo_grsc.getValue()},{myo_cv.getValue()}", file=cost_file)
            if day >= warm_up:
                rp_disc = myo_cost.getValue() + gam*rp_disc

            # Save Action
            action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
            for i in itertools.product(T,M,D,K,C): action['sc'][i] = round(myv_ac_sc[i].X,0) 
            for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = round(myv_ac_rsc[i].X,0)
            for i in itertools.product(T,P): action['uv'][i] = myv_aux_uv[i].X
            for i in PCO: action['ulp'][i] = round(myv_aux_ulp[i].X,0)
            for i in itertools.product(T,P): action['uup'][i] = myv_aux_uup[i].X
            for i in itertools.product(M,D,K,C): action['pwp'][i] = round(myv_aux_pwp[i].X,0)
            for i in itertools.product(T,M,D,K,C): action['psp'][i] = round(myv_aux_psp[i].X,0)
            
            # Log Utilization
            for t in T:
                print(f"{repl},{day},{t-1},{action['uup'][(t,P[0])]},{action['uup'][(t,P[1])]}", file=util_file)

            # Log State / Action
            for m,d,k,c in itertools.product(M,D,K,C): 
                if state['pw'][(m,d,k,c)] != 0: print(f"{repl},{day},state,pw,t,tp,{m},{d},{k},{c},p,{state['pw'][(m,d,k,c)]}", file=logging_file)
            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if state['ps'][(t,m,d,k,c)] != 0: print(f"{repl},{day},state,ps,{t},tp,{m},{d},{k},{c},p,{state['ps'][(t,m,d,k,c)]}", file=logging_file)

            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if action['sc'][(t,m,d,k,c)] != 0: print(f"{repl},{day},action,sc,{t},tp,{m},{d},{k},{c},p,{action['sc'][(t,m,d,k,c)]}", file=logging_file)
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): 
                if action['rsc'][(t,tp,m,d,k,c)] != 0: print(f"{repl},action,rsc,{day},{t},{tp},{m},{d},{k},{c},p,{action['rsc'][(t,tp,m,d,k,c)]}", file=logging_file)
            
            for m,d,k,c in itertools.product(M,D,K,C): 
                if action['pwp'][(m,d,k,c)] != 0: print(f"{repl},{day},post-state,pwp,t,tp,{m},{d},{k},{c},p,{action['pwp'][(m,d,k,c)]}", file=logging_file)
            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if action['psp'][(t,m,d,k,c)] != 0: print(f"{repl},{day},post-state,psp,{t},tp,{m},{d},{k},{c},p,{action['psp'][(t,m,d,k,c)]}", file=logging_file)
            
            #region Save Action Data for Logging
            # Add entries for schedulings
            for m,d,k,c in itertools.product(M, D, K, C):

                skip = True
                for t in T:
                    if action['sc'][(t,m,d,k,c)] > 0.001: skip = False
                if skip == True: continue
                
                # Find all unscheduled patients based on m,d,k,c
                pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)
                
                # Add entry for patients scheduled
                sched = 0
                for t in T:
                    for elem in range(int(action['sc'][(t,m,d,k,c)])):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_unsched['id'].to_list()[sched])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('scheduled')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(t+day-1)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append(pd.NA)
                        sched += 1

            # Add entries for reschedulings
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                
                skip = True
                for tp in T:
                    if action['rsc'][(t,tp,m,d,k,c)] >= 0.001: skip = False
                if skip == True: continue

                pass

                # Finds all scheduled patients based on t,m,d,k,c
                pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                # Add entry for patient rescheduled
                resched = 0
                for tp in T:
                    for elem in range(int(action['rsc'][(t,tp,m,d,k,c)])):  
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[resched])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('rescheduled')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(t+day-1)
                        patient_data['resch_to'].append(tp+day-1)
                        patient_data['transition'].append(pd.NA)
                        resched += 1

                # with open(export_state_my, 'w') as text_file:
                # text_file.write(f'{non_zero_action(action)}\n')
                # text_file.write(f'\tCost: {myo_cost.getValue()}\n')

            # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
            patient_data = {k : [] for k in patient_data}
            #endregion

            # Transition between States
            # Units Leftover / Unit Deviation
            for p in PCO: state['ul'][p] = action['ulp'][p] + round(my_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
            for p in PNCO: state['ul'][p] = round(my_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

            # Patients Waiting - set to post action
            for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = action['pwp'][(m,d,k,c)]

            #region Patients Waiting - calculate & execute D Transition
            my_ptw_d = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    my_ptw_d[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )

            # Save data on transitions
            for m,d,k,c in itertools.product(M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    if my_ptw_d[(m,d,k,c)] == 0: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                    # Save Entry
                    for transition in range(my_ptw_d[(m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(D[D.index(d)+1])
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('complexity')
                        
                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
            
            for m,d,k,c in itertools.product(M, D, K, C): 
                if d != D[-1] and m >= TL[c]: 
                    state['pw'][(m,D[D.index(d)+1],k,c)] = state['pw'][(m,D[D.index(d)+1],k,c)] + my_ptw_d[(m,d,k,c)]
                    state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_d[(m,d,k,c)]
            #endregion

            #region Patients Waiting - calculate & execute K Transition
            my_ptw_k = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    my_ptw_k[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )

            # Save data on transitions
            for m,d,k,c in itertools.product(M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    if my_ptw_k[(m,d,k,c)] == 0: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                    # Save Entry
                    for transition in range(my_ptw_k[(m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                        patient_data['priority'].append(K[K.index(k)+1])
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('priority')
                        
                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
                
            for m,d,k,c in itertools.product(M, D, K, C): 
                if k != K[-1] and m >= TL[c]: 
                    state['pw'][(m,d,K[K.index(k)+1],c)] = state['pw'][(m,d,K[K.index(k)+1],c)] + my_ptw_k[(m,d,k,c)]
                    state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_k[(m,d,k,c)]    
            #endregion

            # Patients Waiting - change wait time
            for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
            for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]

            #region Patient Arrivals
            for d,k,c in itertools.product(D, K, C): 
                arrivals = my_strm_arr.poisson(pea[(d,k,c)])

                # Save Data for Logging
                for arr in range(arrivals):
                    patient_data['repl'].append(repl)
                    patient_data['period'].append(day+1)
                    patient_data['policy'].append('myopic')
                    patient_data['id'].append(patient_id_count)
                    patient_data['priority'].append(k)
                    patient_data['complexity'].append(d)
                    patient_data['surgery'].append(c)
                    patient_data['action'].append('arrived')
                    patient_data['arrived_on'].append(day+1)
                    patient_data['sched_to'].append(pd.NA)
                    patient_data['resch_from'].append(pd.NA)
                    patient_data['resch_to'].append(pd.NA)
                    patient_data['transition'].append(pd.NA)
                    patient_id_count += 1
                state['pw'][(0,d,k,c)] = arrivals
            #endregion

            # Patients Scheduled - post action
            for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = action['psp'][(t,m,d,k,c)]

            #region Patients Scheduled - calculate & execute D Transition
            my_pts_d = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    my_pts_d[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )

            # Save data on transitions
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    if my_pts_d[(t,m,d,k,c)] == 0: continue
                    
                    # Find all scheduled patients based on t,m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Save Entry
                    for transition in range(my_pts_d[(t,m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(D[D.index(d)+1])
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('complexity')

                # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
            patient_data = {k : [] for k in patient_data}

            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if d != D[-1] and m >= TL[c]: 
                    state['ps'][(t,m,D[D.index(d)+1],k,c)] = state['ps'][(t,m,D[D.index(d)+1],k,c)] + my_pts_d[(t,m,d,k,c)]
                    state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_d[(t,m,d,k,c)]
            #endregion

            #region Patients Scheduled - calculate & execute K Transition
            my_pts_k = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    my_pts_k[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )

            # Save data on transitions
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    if my_pts_k[(t,m,d,k,c)] == 0: continue
                    
                    # Find all scheduled patients based on t,m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Save Entry
                    for transition in range(my_pts_k[(t,m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('myopic')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                        patient_data['priority'].append(K[K.index(k)+1])
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('priority')

                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
                        
            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if k != K[-1] and m >= TL[c]: 
                    state['ps'][(t,m,d,K[K.index(k)+1],c)] = state['ps'][(t,m,d,K[K.index(k)+1],c)] + my_pts_k[(t,m,d,k,c)]
                    state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_k[(t,m,d,k,c)]
            # endregion

            # Patients Scheduled  - change wait time
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
            for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
            
            # Patients Scheduled  - change scheduled time
            for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
            for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0

            # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])

        # Save logging
        patient_data_df.to_csv(text_file, index=None,)
            
        my_sim_st.append(rp_st)
        my_sim_ac.append(rp_ac)
        my_sim_cost.append(rp_cost)
        my_sim_disc.append(rp_disc)

    #endregion

    ##### MDP Simulation
    #region

    # Simulation Data
    md_sim_st = []
    md_sim_ac = []
    md_sim_cost = []
    md_sim_disc = []

    text_file = open(export_state_md, 'w', newline="")
    logging_file = open(export_sa_md, 'w', newline="")
    cost_file = open(export_cost_md, 'w', newline="")
    util_file = open(export_util_md, 'w', newline="")
    print(f"repl,period,cost,cost_cw,cost_cs,cost_brsc,cost_grsc,cost_cv", file=cost_file)
    print(f"repl,period,horizon_period,usage_admin,usage_OR", file=util_file)
    print(f"repl,period,state-aciton,value,t,tp,m,d,k,c,p,val", file=logging_file)

    # Simulation
    for repl in trange(replications, desc='MDP'):
        
        # Random streams
        md_strm_dev = np.random.default_rng(repl)
        md_strm_pwt = np.random.default_rng(repl)
        md_strm_pst = np.random.default_rng(repl)
        md_strm_arr = np.random.default_rng(repl)
            
        # Replication Data
        rp_st = []
        rp_ac = []
        rp_cost = []
        rp_disc = 0
        state = deepcopy(init_state)
        
        # Detailed Logging Data
        patient_data_df = pd.DataFrame( columns=['repl', 'period', 'policy', 'id','priority', 'complexity', 'surgery','action','arrived_on','sched_to', 'resch_from', 'resch_to', 'transition'] )
        patient_id_count = 0

        for day in trange(duration):

            # Init Patient Data
            patient_data = {'repl': [], 'period': [], 'policy': [], 'id': [], 'priority': [], 'complexity': [], 'surgery': [], 'action': [], 'arrived_on': [], 'sched_to': [], 'resch_from': [], 'resch_to': [], 'transition': []}
            if day == 0:
                for m,d,k,c in itertools.product(M,D,K,C):
                    for elem in range(int(state['pw'][(m,d,k,c)])):  
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(patient_id_count)
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('arrived')
                        patient_data['arrived_on'].append(day-m)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append(pd.NA)
                        patient_id_count += 1     
                patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                patient_data = {k : [] for k in patient_data}

            # Generate Action
            if day >= warm_up:
                for p in P: mdv_st_ul[p].UB = state['ul'][p]; mdv_st_ul[p].LB = state['ul'][p]
                for m,d,k,c in itertools.product(M, D, K, C): mdv_st_pw[(m,d,k,c)].UB = state['pw'][(m,d,k,c)]; mdv_st_pw[(m,d,k,c)].LB = state['pw'][(m,d,k,c)]
                for t,m,d,k,c in itertools.product(T, M, D, K, C): mdv_st_ps[(t,m,d,k,c)].UB = state['ps'][(t,m,d,k,c)]; mdv_st_ps[(t,m,d,k,c)].LB = state['ps'][(t,m,d,k,c)]
                MDP.optimize()
            else:

                for k in K: myv_cost_cw[k].UB = cv-1; myv_cost_cw[k].LB = cv-1;
                for p in P: myv_st_ul[p].UB = state['ul'][p]; myv_st_ul[p].LB = state['ul'][p]
                for m,d,k,c in itertools.product(M, D, K, C): myv_st_pw[(m,d,k,c)].UB = round(state['pw'][(m,d,k,c)],0); myv_st_pw[(m,d,k,c)].LB = round(state['pw'][(m,d,k,c)],0)
                for t,m,d,k,c in itertools.product(T, M, D, K, C): myv_st_ps[(t,m,d,k,c)].UB = round(state['ps'][(t,m,d,k,c)],0); myv_st_ps[(t,m,d,k,c)].LB = round(state['ps'][(t,m,d,k,c)],0)
                myopic.optimize()

            # Save Cost
            if day >= warm_up:
                rp_cost.append(mdo_cost.getValue())
                print(f"{repl},{day}, {mdo_cost.getValue()},{mdo_cw.getValue()},{mdo_cs.getValue()},{mdo_brsc.getValue()},{mdo_grsc.getValue()},{mdo_cv.getValue()}", file=cost_file)
                rp_disc = mdo_cost.getValue() + gam*rp_disc
            else:
                for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
                for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = round(myv_ac_sc[(t,m,d,k,c)].X,0); myv_ac_sc[(t,m,d,k,c)].LB = round(myv_ac_sc[(t,m,d,k,c)].X,0)
                for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = round(myv_ac_rsc[(t,tp,m,d,k,c)].X,0); myv_ac_rsc[(t,tp,m,d,k,c)].LB = round(myv_ac_rsc[(t,tp,m,d,k,c)].X,0)
                myopic.optimize()
                rp_cost.append(myo_cost.getValue())
                print(f"{repl},{day}, {myo_cost.getValue()},{myo_cw.getValue()},{myo_cs.getValue()},{myo_brsc.getValue()},{myo_grsc.getValue()},{myo_cv.getValue()}", file=cost_file)

            # Save Action
            if day >= warm_up:
                action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                for i in itertools.product(T,M,D,K,C): action['sc'][i] = round(mdv_ac_sc[i].X,0)
                for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = round(mdv_ac_rsc[i].X,0)
                for i in itertools.product(T,P): action['uv'][i] = mdv_aux_uv[i].X
                for i in PCO: action['ulp'][i] = round(mdv_aux_ulp[i].X,0)
                for i in itertools.product(T,P): action['uup'][i] = mdv_aux_uup[i].X
                for i in itertools.product(M,D,K,C): action['pwp'][i] = round(mdv_aux_pwp[i].X,0)
                for i in itertools.product(T,M,D,K,C): action['psp'][i] = round(mdv_aux_psp[i].X,0)

                for t in T:
                    print(f"{repl},{day},{t-1},{action['uup'][(t,P[0])]},{action['uup'][(t,P[1])]}", file=util_file)
            else:
                
                for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = GRB.INFINITY; myv_ac_sc[(t,m,d,k,c)].LB = 0
                for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = GRB.INFINITY; myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0
                action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                for i in itertools.product(T,M,D,K,C): action['sc'][i] = round(myv_ac_sc[i].X,0) 
                for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = round(myv_ac_rsc[i].X,0)
                for i in itertools.product(T,P): action['uv'][i] = myv_aux_uv[i].X
                for i in PCO: action['ulp'][i] = round(myv_aux_ulp[i].X,0)
                for i in itertools.product(T,P): action['uup'][i] = myv_aux_uup[i].X
                for i in itertools.product(M,D,K,C): action['pwp'][i] = round(myv_aux_pwp[i].X,0)
                for i in itertools.product(T,M,D,K,C): action['psp'][i] = round(myv_aux_psp[i].X,0)
                
                for t in T:
                    print(f"{repl},{day},{t-1},{action['uup'][(t,P[0])]},{action['uup'][(t,P[1])]}", file=util_file)

            # Log State / Action
            for m,d,k,c in itertools.product(M,D,K,C): 
                if state['pw'][(m,d,k,c)] != 0: print(f"{repl},{day},state,pw,t,tp,{m},{d},{k},{c},p,{state['pw'][(m,d,k,c)]}", file=logging_file)
            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if state['ps'][(t,m,d,k,c)] != 0: print(f"{repl},{day},state,ps,{t},tp,{m},{d},{k},{c},p,{state['ps'][(t,m,d,k,c)]}", file=logging_file)

            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if action['sc'][(t,m,d,k,c)] != 0: print(f"{repl},{day},action,sc,{t},tp,{m},{d},{k},{c},p,{action['sc'][(t,m,d,k,c)]}", file=logging_file)
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): 
                if action['rsc'][(t,tp,m,d,k,c)] != 0: print(f"{repl},action,rsc,{day},{t},{tp},{m},{d},{k},{c},p,{action['rsc'][(t,tp,m,d,k,c)]}", file=logging_file)
            
            for m,d,k,c in itertools.product(M,D,K,C): 
                if action['pwp'][(m,d,k,c)] != 0: print(f"{repl},{day},post-state,pwp,t,tp,{m},{d},{k},{c},p,{action['pwp'][(m,d,k,c)]}", file=logging_file)
            for t,m,d,k,c in itertools.product(T,M,D,K,C): 
                if action['psp'][(t,m,d,k,c)] != 0: print(f"{repl},{day},post-state,psp,{t},tp,{m},{d},{k},{c},p,{action['psp'][(t,m,d,k,c)]}", file=logging_file)

            #region Save Action Data for Logging
            # Add entries for schedulings
            for m,d,k,c in itertools.product(M, D, K, C):

                skip = True
                for t in T:
                    if action['sc'][(t,m,d,k,c)] > 0.001: skip = False
                if skip == True: continue
                
                # Find all unscheduled patients based on m,d,k,c
                pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)
                
                # Add entry for patients scheduled
                sched = 0
                for t in T:
                    for elem in range(int(action['sc'][(t,m,d,k,c)])):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_unsched['id'].to_list()[sched])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('scheduled')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(t+day-1)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append(pd.NA)
                        sched += 1

            # Add entries for reschedulings
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                
                skip = True
                for tp in T: 
                    if action['rsc'][(t,tp,m,d,k,c)] >= 0.001: skip = False
                if skip == True: continue

                # Finds all scheduled patients based on t,m,d,k,c
                pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                # Add entry for patient rescheduled
                resched = 0
                for tp in T:
                    for elem in range(int(action['rsc'][(t,tp,m,d,k,c)])):  
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[resched])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('rescheduled')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(t+day-1)
                        patient_data['resch_to'].append(tp+day-1)
                        patient_data['transition'].append(pd.NA)
                        resched += 1

            # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
            patient_data = {k : [] for k in patient_data}
            #endregion
            
            ### Transition between States
            # Units Leftover / Unit Deviation   
            for p in PCO: state['ul'][p] = action['ulp'][p] + round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
            for p in PNCO: state['ul'][p] = round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

            # Patients Waiting - set to post action
            for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = action['pwp'][(m,d,k,c)]

            #region Patients Waiting - calculate & execute D Transition
            md_ptw_d = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    md_ptw_d[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )

            # Save data on transitions
            for m,d,k,c in itertools.product(M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    if md_ptw_d[(m,d,k,c)] == 0: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                    # Save Entry
                    for transition in range(md_ptw_d[(m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(D[D.index(d)+1])
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('complexity')
                        
                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
            
            for m,d,k,c in itertools.product(M, D, K, C): 
                if d != D[-1] and m >= TL[c]: 
                    state['pw'][(m,D[D.index(d)+1],k,c)] = state['pw'][(m,D[D.index(d)+1],k,c)] + md_ptw_d[(m,d,k,c)]
                    state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - md_ptw_d[(m,d,k,c)]
            #endregion

            #region Patients Waiting - calculate & execute K Transition
            md_ptw_k = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    md_ptw_k[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )

            # Save data on transitions
            for m,d,k,c in itertools.product(M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    if md_ptw_k[(m,d,k,c)] == 0: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                    # Save Entry
                    for transition in range(md_ptw_k[(m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                        patient_data['priority'].append(K[K.index(k)+1])
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('priority')
                        
                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
                
            for m,d,k,c in itertools.product(M, D, K, C): 
                if k != K[-1] and m >= TL[c]: 
                    state['pw'][(m,d,K[K.index(k)+1],c)] = state['pw'][(m,d,K[K.index(k)+1],c)] + md_ptw_k[(m,d,k,c)]
                    state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - md_ptw_k[(m,d,k,c)]    
            #endregion

            # Patients Waiting - change wait time
            for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
            for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]

            #region Patient Arrivals
            for d,k,c in itertools.product(D, K, C): 
                arrivals = md_strm_arr.poisson(pea[(d,k,c)])

                # Save Data for Logging
                for arr in range(arrivals):
                    patient_data['repl'].append(repl)
                    patient_data['period'].append(day+1)
                    patient_data['policy'].append('MDP')
                    patient_data['id'].append(patient_id_count)
                    patient_data['priority'].append(k)
                    patient_data['complexity'].append(d)
                    patient_data['surgery'].append(c)
                    patient_data['action'].append('arrived')
                    patient_data['arrived_on'].append(day+1)
                    patient_data['sched_to'].append(pd.NA)
                    patient_data['resch_from'].append(pd.NA)
                    patient_data['resch_to'].append(pd.NA)
                    patient_data['transition'].append(pd.NA)
                    patient_id_count += 1
                state['pw'][(0,d,k,c)] = arrivals
            #endregion

            # Patients Scheduled - post action
            for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = action['psp'][(t,m,d,k,c)]

            #region Patients Scheduled - calculate & execute D Transition
            md_pts_d = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    md_pts_d[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )

            # Save data on transitions
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if d != D[-1] and m >= TL[c]: 
                    if md_pts_d[(t,m,d,k,c)] == 0: continue
                    
                    # Find all scheduled patients based on t,m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Save Entry
                    for transition in range(md_pts_d[(t,m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                        patient_data['priority'].append(k)
                        patient_data['complexity'].append(D[D.index(d)+1])
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('complexity')

                # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
            patient_data = {k : [] for k in patient_data}

            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if d != D[-1] and m >= TL[c]: 
                    state['ps'][(t,m,D[D.index(d)+1],k,c)] = state['ps'][(t,m,D[D.index(d)+1],k,c)] + md_pts_d[(t,m,d,k,c)]
                    state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - md_pts_d[(t,m,d,k,c)]
            #endregion

            #region Patients Scheduled - calculate & execute K Transition
            md_pts_k = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    md_pts_k[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )

            # Save data on transitions
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if k != K[-1] and m >= TL[c]: 
                    if md_pts_k[(t,m,d,k,c)] == 0: continue
                    
                    # Find all scheduled patients based on t,m,d,k,c
                    pat_subset = retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Save Entry
                    for transition in range(md_pts_k[(t,m,d,k,c)]):
                        patient_data['repl'].append(repl)
                        patient_data['period'].append(day)
                        patient_data['policy'].append('MDP')
                        patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                        patient_data['priority'].append(K[K.index(k)+1])
                        patient_data['complexity'].append(d)
                        patient_data['surgery'].append(c)
                        patient_data['action'].append('transition')
                        patient_data['arrived_on'].append(pd.NA)
                        patient_data['sched_to'].append(pd.NA)
                        patient_data['resch_from'].append(pd.NA)
                        patient_data['resch_to'].append(pd.NA)
                        patient_data['transition'].append('priority')

                    # Save Data to dataframe
                    patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                    patient_data = {k : [] for k in patient_data}
                        
            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if k != K[-1] and m >= TL[c]: 
                    state['ps'][(t,m,d,K[K.index(k)+1],c)] = state['ps'][(t,m,d,K[K.index(k)+1],c)] + md_pts_k[(t,m,d,k,c)]
                    state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - md_pts_k[(t,m,d,k,c)]
            # endregion

            # Patients Scheduled  - change wait time
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
            for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
            
            # Patients Scheduled  - change scheduled time
            for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
            for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0

            # Save Data to dataframe
            patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
        
        # Save logging
        patient_data_df.to_csv(text_file, index=None,)

        md_sim_st.append(rp_st)
        md_sim_ac.append(rp_ac)
        md_sim_cost.append(rp_cost)
        md_sim_disc.append(rp_disc)

    #endregion

    ##### Reporting #####
    #region
    my_avg_cost = np.average(np.transpose(my_sim_cost),axis=1)
    md_avg_cost = np.average(np.transpose(md_sim_cost),axis=1)
    x_dat = [i for i in range(len(my_avg_cost))]

    # Plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Myopic', 'MDP'))
    fig.add_trace( go.Line(x=x_dat, y=my_avg_cost), row=1, col=1 )
    fig.add_trace( go.Line(x=x_dat, y=md_avg_cost), row=1, col=2 )
    fig.write_html(os.path.join(my_path, export_pic))
    # fig.show()

    # Print Cost
    with open(os.path.join(my_path, export_txt), "w") as text_file:
        print(f'Myopic: \t Discounted Cost {np.average(my_sim_disc):.2f} \t Average Cost {np.average(my_avg_cost[warm_up:]):.2f} \t Warm Up Cost {np.average(my_avg_cost[:warm_up]):.2f}', file=text_file)
        print(f'MDP: \t\t Discounted Cost {np.average(md_sim_disc):.2f} \t Average Cost {np.average(md_avg_cost[warm_up:]):.2f} \t Warm Up Cost {np.average(md_avg_cost[:warm_up]):.2f}', file=text_file)