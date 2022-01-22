#%%
##### Initialization & Changeable Parameters #####
#region
from os import linesep

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

#endregion

def main_func(replications, warm_up, duration, show_policy, import_data, import_beta, export_txt, export_pic):
    ##### Read Data #####
    #region

    # replications = 3
    # warm_up = 1000
    # duration = 3000
    # show_policy =  False
    # import_data = "Data/sens-data/simple/cw1-cc1-cv100-gam95-simple-data.xlsx"
    # import_beta = "Data/sens-data/simple/betas/cw1-cc1-cv100-gam95-simple-optimal.pkl"
    # export_txt = "Data/sens-res/simple/cw1-cc1-cv100-gam95-simple-optimal-res.txt"
    # export_pic = "Data/sens-res/simple/cw1-cc1-cv100-gam95-simple-optimal-res.html"

    my_path = os.getcwd()
    input_data = data_import.read_data(os.path.join(my_path, import_data))

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
    init_state_strm =  np.random.default_rng(199725)
    for i in itertools.product(M,D,K,C): init_state['pw'][i] = init_state_strm.poisson(E_PW[i])
    for i in itertools.product(T,M,D,K,C): init_state['ps'][i] = init_state_strm.poisson(E_PS[i])

    # Betas
    with open(os.path.join(my_path, import_beta), 'rb') as handle:
        betas = pickle.load(handle)
    #endregion

    ##### Various Functions #####
    #region
    def non_zero_state(state):
        non_zero_st = {'ul': {}, 'pw': {}, 'ps': {}}
        for k,v in state['ul'].items(): 
            if v != 0: 
                print(f'\tState - Units Left Over - {(k)} - {v}')
                non_zero_st['ul'][k] = v
        for k,v in state['pw'].items(): 
            if v != 0: 
                print(f'\tState - Patients Waiting - {k} - {v}')
                non_zero_st['pw'][k] = v
        for k,v in state['ps'].items(): 
            if v != 0: 
                print(f'\tState - Patients Scheduled - {k} - {v}')
                non_zero_st['ps'][k] = v
        return(non_zero_st)

    def non_zero_action(action):
        non_zero_ac = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
        for k,v in action['sc'].items():
            if v != 0: 
                print(f'\tAction - Schedule Patients - {k} - {v}')
                non_zero_ac['sc'][k] = v
        for k,v in action['rsc'].items():
            if v != 0: 
                print(f'\tAction - Reschedule Patients - {k} - {v}')
                non_zero_ac['rsc'][k] = v
        for k,v in action['uv'].items():
            if v != 0: 
                print(f'\tAction - Units Violated - {k} - {v}')
                non_zero_ac['uv'][k] = v
        for k,v in action['ulp'].items():
            if v != 0: 
                print(f'\tPost Action - Units Left Over - {k} - {v}')
                non_zero_ac['ulp'][k] = v
        for k,v in action['pwp'].items():
            if v != 0: 
                print(f'\tPost Action - Patients Waiting - {k} - {v}')
                non_zero_ac['pwp'][k] = v
        for k,v in action['psp'].items():
            if v != 0: 
                print(f'\tPost Action - Patients Scheduled - {k} - {v}')
                non_zero_ac['psp'][k] = v
        for k,v in action['uup'].items():
            if v != 0: 
                print(f'\tPost Action - Units Used - {k} - {v}')
                non_zero_ac['uup'][k] = v
        return(non_zero_ac)
    #endregion

    ##### Myopic Model #####
    #region
    myopic = Model('Myopic')
    myopic.params.LogToConsole = 0

    # Cost Params
    myv_cost_cw = myopic.addVars(K, vtype=GRB.CONTINUOUS, name='var_cost_cw')
    myv_cost_cs = myopic.addVars(K,M, vtype=GRB.CONTINUOUS, name='var_cost_cs')
    myv_cost_cc = myopic.addVars(K, vtype=GRB.CONTINUOUS, name='var_cost_cc')
    myv_cost_cv = myopic.addVar(vtype=GRB.CONTINUOUS, name='var_cost_cv')

    # Fix Costs
    for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
    for m,k in itertools.product(M, K): myv_cost_cs[(k,m)].UB = cs[k][m]; myv_cost_cs[(k,m)].LB = cs[k][m];
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
    MDP = Model('Myopic')
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
    #%%
    ##### Myopic Simulation
    #region

    # Simulation Data
    my_sim_st = []
    my_sim_ac = []
    my_sim_cost = []
    my_sim_disc = []

    # Simulation
    for repl in trange(replications, desc='Myopic'):

        # Random streams
        my_strm_dev = np.random.default_rng(repl)
        my_strm_pwt = np.random.default_rng(repl)
        my_strm_pst = np.random.default_rng(repl)
        my_strm_arr = np.random.default_rng(repl)
            
        # Replication Data
        rp_st = []
        rp_ac = []
        rp_cost = []
        rp_disc = 0
        state = deepcopy(init_state)

        # Single Replication
        for day in range(duration):

            # Save State Data
            if show_policy: rp_st.append(deepcopy(state))

            # Generate Action (With slightly different logic)
            for k in K: myv_cost_cw[k].UB = cv-1; myv_cost_cw[k].LB = cv-1;
            for p in P: myv_st_ul[p].UB = state['ul'][p]; myv_st_ul[p].LB = state['ul'][p]
            for m,d,k,c in itertools.product(M, D, K, C): myv_st_pw[(m,d,k,c)].UB = state['pw'][(m,d,k,c)]; myv_st_pw[(m,d,k,c)].LB = state['pw'][(m,d,k,c)]
            for t,m,d,k,c in itertools.product(T, M, D, K, C): myv_st_ps[(t,m,d,k,c)].UB = state['ps'][(t,m,d,k,c)]; myv_st_ps[(t,m,d,k,c)].LB = state['ps'][(t,m,d,k,c)]
            myopic.optimize()

            # Save Cost (with normal logic)
            for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
            for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = myv_ac_sc[(t,m,d,k,c)].X; myv_ac_sc[(t,m,d,k,c)].LB = myv_ac_sc[(t,m,d,k,c)].X
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = myv_ac_rsc[(t,tp,m,d,k,c)].X; myv_ac_rsc[(t,tp,m,d,k,c)].LB = myv_ac_rsc[(t,tp,m,d,k,c)].X
            myopic.optimize()

            # Reset Myopic
            for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = GRB.INFINITY; myv_ac_sc[(t,m,d,k,c)].LB = 0
            for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = GRB.INFINITY; myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0

            rp_cost.append(myo_cost.getValue())
            if day >= warm_up:
                rp_disc = myo_cost.getValue() + gam*rp_disc

            # Save Action
            action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
            for i in itertools.product(T,M,D,K,C): action['sc'][i] = myv_ac_sc[i].X 
            for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = myv_ac_rsc[i].X
            for i in itertools.product(T,P): action['uv'][i] = myv_aux_uv[i].X
            for i in PCO: action['ulp'][i] = myv_aux_ulp[i].X
            for i in itertools.product(T,P): action['uup'][i] = myv_aux_uup[i].X
            for i in itertools.product(M,D,K,C): action['pwp'][i] = myv_aux_pwp[i].X
            for i in itertools.product(T,M,D,K,C): action['psp'][i] = myv_aux_psp[i].X
            if show_policy: rp_ac.append(action)

            # Transition between States
            # Units Leftover / Unit Deviation
            for p in PCO: state['ul'][p] = myv_aux_ulp[p].X + round(my_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
            for p in PNCO: state['ul'][p] = round(my_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

            # Patients Waiting - set to post action
            for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = myv_aux_pwp[(m,d,k,c)].X
            # Patients Waiting - calculate & execute D Transition
            my_ptw_d = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if d != D[-1] and m >= TL[c]: my_ptw_d[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )
            for m,d,k,c in itertools.product(M, D, K, C): 
                if d != D[0] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] + my_ptw_d[(m,D[D.index(d)-1],k,c)]
            # Patients Waiting - calculate & execute K Transition
            my_ptw_k = {}
            for m,d,k,c in itertools.product(M, D, K, C):
                if k != K[-1] and m >= TL[c]: my_ptw_k[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )
            for m,d,k,c in itertools.product(M, D, K, C): 
                if d != D[-1] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_d[(m,d,k,c)]      
            # Patients Waiting - change wait time
            for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
            for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]
            for d,k,c in itertools.product(D, K, C): state['pw'][(0,d,k,c)] = my_strm_arr.poisson(pea[(d,k,c)])

            # Patients Scheduled - post action
            for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = myv_aux_psp[(t,m,d,k,c)].X
            # Patients Scheduled - calculate & execute D Transition
            my_pts_d = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if d != D[-1] and m >= TL[c]: my_pts_d[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )
            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if d != D[0] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] + my_pts_d[(t,m,D[D.index(d)-1],k,c)]
            # Patients Scheduled - calculate & execute K Transition
            my_pts_k = {}
            for t,m,d,k,c in itertools.product(T, M, D, K, C):
                if k != K[-1] and m >= TL[c]: my_pts_k[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )
            for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                if d != D[-1] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_d[(t,m,d,k,c)]     
            # Patients Scheduled  - change wait time
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
            for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
            for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
            # Patients Scheduled  - change scheduled time
            for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
            for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0

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

        for day in range(duration):

            # Save State Data
            if show_policy: rp_st.append(deepcopy(state))

            # Generate Action
            if day >= warm_up:
                for p in P: mdv_st_ul[p].UB = state['ul'][p]; mdv_st_ul[p].LB = state['ul'][p]
                for m,d,k,c in itertools.product(M, D, K, C): mdv_st_pw[(m,d,k,c)].UB = state['pw'][(m,d,k,c)]; mdv_st_pw[(m,d,k,c)].LB = state['pw'][(m,d,k,c)]
                for t,m,d,k,c in itertools.product(T, M, D, K, C): mdv_st_ps[(t,m,d,k,c)].UB = state['ps'][(t,m,d,k,c)]; mdv_st_ps[(t,m,d,k,c)].LB = state['ps'][(t,m,d,k,c)]
                MDP.optimize()
            else:

                for k in K: myv_cost_cw[k].UB = cv-1; myv_cost_cw[k].LB = cv-1;
                for p in P: myv_st_ul[p].UB = state['ul'][p]; myv_st_ul[p].LB = state['ul'][p]
                for m,d,k,c in itertools.product(M, D, K, C): myv_st_pw[(m,d,k,c)].UB = state['pw'][(m,d,k,c)]; myv_st_pw[(m,d,k,c)].LB = state['pw'][(m,d,k,c)]
                for t,m,d,k,c in itertools.product(T, M, D, K, C): myv_st_ps[(t,m,d,k,c)].UB = state['ps'][(t,m,d,k,c)]; myv_st_ps[(t,m,d,k,c)].LB = state['ps'][(t,m,d,k,c)]
                myopic.optimize()

            # Save Cost
            if day >= warm_up:
                rp_cost.append(mdo_cost.getValue())
                rp_disc = mdo_cost.getValue() + gam*rp_disc
            else:
                for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];
                for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = myv_ac_sc[(t,m,d,k,c)].X; myv_ac_sc[(t,m,d,k,c)].LB = myv_ac_sc[(t,m,d,k,c)].X
                for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = myv_ac_rsc[(t,tp,m,d,k,c)].X; myv_ac_rsc[(t,tp,m,d,k,c)].LB = myv_ac_rsc[(t,tp,m,d,k,c)].X
                myopic.optimize()
                rp_cost.append(myo_cost.getValue())

            # Save Action
            if day >= warm_up:
                action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                for i in itertools.product(T,M,D,K,C): action['sc'][i] = mdv_ac_sc[i].X 
                for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = mdv_ac_rsc[i].X
                for i in itertools.product(T,P): action['uv'][i] = mdv_aux_uv[i].X
                for i in PCO: action['ulp'][i] = mdv_aux_ulp[i].X
                for i in itertools.product(T,P): action['uup'][i] = mdv_aux_uup[i].X
                for i in itertools.product(M,D,K,C): action['pwp'][i] = mdv_aux_pwp[i].X
                for i in itertools.product(T,M,D,K,C): action['psp'][i] = mdv_aux_psp[i].X
                if show_policy: rp_ac.append(action)
            else:
                
                for t,m,d,k,c in itertools.product(T,M,D,K,C): myv_ac_sc[(t,m,d,k,c)].UB = GRB.INFINITY; myv_ac_sc[(t,m,d,k,c)].LB = 0
                for t,tp,m,d,k,c in itertools.product(T,T,M,D,K,C): myv_ac_rsc[(t,tp,m,d,k,c)].UB = GRB.INFINITY; myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0
                action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                for i in itertools.product(T,M,D,K,C): action['sc'][i] = myv_ac_sc[i].X 
                for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = myv_ac_rsc[i].X
                for i in itertools.product(T,P): action['uv'][i] = myv_aux_uv[i].X
                for i in PCO: action['ulp'][i] = myv_aux_ulp[i].X
                for i in itertools.product(T,P): action['uup'][i] = myv_aux_uup[i].X
                for i in itertools.product(M,D,K,C): action['pwp'][i] = myv_aux_pwp[i].X
                for i in itertools.product(T,M,D,K,C): action['psp'][i] = myv_aux_psp[i].X
                if show_policy: rp_ac.append(action)
            
            # Transition between States
            if day >= warm_up:
                # Units Leftover / Unit Deviation
                for p in PCO: state['ul'][p] = mdv_aux_ulp[p].X + round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
                for p in PNCO: state['ul'][p] = round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

                # Patients Waiting - set to post action
                for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = mdv_aux_pwp[(m,d,k,c)].X
                # Patients Waiting - calculate D&K transition
                md_ptw_d = {}
                md_ptw_k = {}
                for m,d,k,c in itertools.product(M, D, K, C):
                    if d != D[-1] and m >= TL[c]: md_ptw_d[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )
                    if k != K[-1] and m >= TL[c]: md_ptw_k[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )
                for m,d,k,c in itertools.product(M, D, K, C): 
                    if d != D[0] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] + md_ptw_d[(m,D[D.index(d)-1],k,c)]
                    if d != D[-1] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - md_ptw_d[(m,d,k,c)]      
                # Patients Waiting - change wait time
                for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
                for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]
                for d,k,c in itertools.product(D, K, C): state['pw'][(0,d,k,c)] = md_strm_arr.poisson(pea[(d,k,c)])

                # Patients Scheduled - post action
                for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = mdv_aux_psp[(t,m,d,k,c)].X
                # Patients Scheduled  - calculate D&K transition
                md_pts_d = {}
                md_pts_k = {}
                for t,m,d,k,c in itertools.product(T, M, D, K, C):
                    if d != D[-1] and m >= TL[c]: md_pts_d[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )
                    if k != K[-1] and m >= TL[c]: md_pts_k[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )
                for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                    if d != D[0] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] + md_pts_d[(t,m,D[D.index(d)-1],k,c)]
                    if d != D[-1] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - md_pts_d[(t,m,d,k,c)]     
                # Patients Scheduled  - change wait time
                for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
                for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
                for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
                # Patients Scheduled  - change scheduled time
                for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
                for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0
                pass
            else:
                # Units Leftover / Unit Deviation
                for p in PCO: state['ul'][p] = myv_aux_ulp[p].X + round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
                for p in PNCO: state['ul'][p] = round(md_strm_dev.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

                # Patients Waiting - set to post action
                for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = myv_aux_pwp[(m,d,k,c)].X
                # Patients Waiting - calculate D&K transition
                my_ptw_d = {}
                my_ptw_k = {}
                for m,d,k,c in itertools.product(M, D, K, C):
                    if d != D[-1] and m >= TL[c]: my_ptw_d[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )
                    if k != K[-1] and m >= TL[c]: my_ptw_k[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )
                for m,d,k,c in itertools.product(M, D, K, C): 
                    if d != D[0] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] + my_ptw_d[(m,D[D.index(d)-1],k,c)]
                    if d != D[-1] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_d[(m,d,k,c)]      
                # Patients Waiting - change wait time
                for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
                for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]
                for d,k,c in itertools.product(D, K, C): state['pw'][(0,d,k,c)] = md_strm_arr.poisson(pea[(d,k,c)])

                # Patients Scheduled - post action
                for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = myv_aux_psp[(t,m,d,k,c)].X
                # Patients Scheduled  - calculate D&K transition
                my_pts_d = {}
                my_pts_k = {}
                for t,m,d,k,c in itertools.product(T, M, D, K, C):
                    if d != D[-1] and m >= TL[c]: my_pts_d[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )
                    if k != K[-1] and m >= TL[c]: my_pts_k[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )
                for t,m,d,k,c in itertools.product(T, M, D, K, C): 
                    if d != D[0] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] + my_pts_d[(t,m,D[D.index(d)-1],k,c)]
                    if d != D[-1] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_d[(t,m,d,k,c)]     
                # Patients Scheduled  - change wait time
                for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
                for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
                for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
                # Patients Scheduled  - change scheduled time
                for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
                for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0
                pass

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
    # print(f'Myopic: \t Discounted Cost {np.average(my_sim_disc):.2f} \t Average Cost {np.average(my_avg_cost[warm_up:]):.2f} \t Warm Up Cost {np.average(my_avg_cost[:warm_up]):.2f}')
    # print(f'MDP: \t\t Discounted Cost {np.average(md_sim_disc):.2f} \t Average Cost {np.average(md_avg_cost[warm_up:]):.2f} \t Warm Up Cost {np.average(md_avg_cost[:warm_up]):.2f}')
    #endregion

    ##### Expected State Values #####
    #region
    if show_policy:
        total_days = replications * (duration - warm_up)
        exp_st = {'ul': {}, 'pw':{}, 'ps': {}}
        for p in P: exp_st['ul'][p] = 0
        for i in itertools.product(M,D,K,C): exp_st['pw'][i] = 0
        for i in itertools.product(T,M,D,K,C): exp_st['ps'][i] = 0

        for repl in range(len(my_sim_st)):
            for day in range(len(my_sim_st[repl][warm_up:])):
                for p in P: exp_st['ul'][p] += my_sim_st[repl][day+warm_up]['ul'][p] / total_days
                for i in itertools.product(M,D,K,C): exp_st['pw'][i] += my_sim_st[repl][day+warm_up]['pw'][i] / total_days
                for i in itertools.product(T,M,D,K,C): exp_st['ps'][i] += my_sim_st[repl][day+warm_up]['ps'][i] / total_days
    #endregion

    ##### Comparison of policies #####
    #region
    if show_policy:
        print('MYOPIC POLICY')
        for i in range(20):
            print(f'Day {i+1}')
            non_zero_state(my_sim_st[0][i])
            non_zero_action(my_sim_ac[0][i])
            print(f'\tCost: {my_sim_cost[0][i]}')
        print()
        
        print('MDP POLICY')
        for i in range(20):
            print(f'Day {i+1}')
            non_zero_state(md_sim_st[0][i])
            non_zero_action(md_sim_ac[0][i])
            print(f'\tCost: {md_sim_cost[0][i]}')
        #endregion
        # %%
