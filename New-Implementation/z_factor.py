#%%
from Modules import data_import

import os.path
import pickle
import itertools

from gurobipy import *
import numpy as np
import pandas as pd

from pprint import pprint
import plotly.express as px

# Improting Data
test_modifier = "cw1-cc5-cv100-gam95-"
data_type = "smaller-full"

import_data =  f"Data/sens-data/{data_type}/{test_modifier}{data_type}-data.xlsx"
import_beta = f"Data/sens-data/{data_type}/betas/{test_modifier}{data_type}-optimal.pkl"

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

# MDP
MDP = Model('MDP')
# MDP.params.LogToConsole = 0

# State Action & Auxiliary Variables
mdv_st_ul = MDP.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, ub=0, name='var_state_ul')
mdv_st_pw = MDP.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=2, ub=2, name='var_state_pw')
mdv_st_ps = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, ub=0, name='var_state_ps')

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
MDP.update()

#%%
# FACTORIZING

sc_coef = {}
for i in itertools.product(T,M,D,K,C):
    sc_coef[i] = 0

# Cost Function
for i in itertools.product(T,M,D,K,C):
    sc_coef[i] += mdv_ac_sc[i].Obj

# ul
for p,m,d,k,c in itertools.product(P, M, D, K, C):
    sc_coef[(1,m,d,k,c)] += betas['bul'][p]

# pw
for t,m,d,k,c in itertools.product(T,M,D,K,C):
    # part 1
    if 1 <= m <= M[-2]: 
        sc_coef[(t,m-1,d,k,c)] -= betas['bpw'][m,d,k,c]
    if m == M[-1]:
        sc_coef[(t,m-1,d,k,c)] -= betas['bpw'][m,d,k,c]
    if m == M[-1]:
        sc_coef[(t,m,d,k,c)] -= betas['bpw'][m,d,k,c]
    
    # part 2
    if (TL[c] <= m <= M[-2]) and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],k,c)] -= betas['bpw'][m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],k,c)] -= betas['bpw'][m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and d != D[0]:
        sc_coef[(t,m,D[D.index(d)-1],k,c)] -= betas['bpw'][m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    
    # part 3
    if (TL[c] <= m <= M[-2]) and d != D[-1]:
        sc_coef[(t,m-1,d,k,c)] += betas['bpw'][m,d,k,c] * ptp_d[(d,c)]
    if m == M[-1] and d != D[0]:
        sc_coef[(t,m-1,d,k,c)] += betas['bpw'][m,d,k,c] * ptp_d[(d,c)]
    if m == M[-1] and d != D[0]:
        sc_coef[(t,m,d,k,c)] += betas['bpw'][m,d,k,c] * ptp_d[(d,c)]

    # part 4
    if (TL[c] <= m <= M[-2]) and k != K[0]:
        sc_coef[(t,m-1,d,K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if (TL[c] <= m <= M[-2]) and k != K[0] and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if (TL[c] <= m <= M[-2]) and k != K[0] and d != D[-1]:
        sc_coef[(t,m-1,d,K[K.index(k)-1],c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    if m == M[-1] and k != K[0]:
        sc_coef[(t,m-1,d,K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if m == M[-1] and k != K[0] and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and k != K[0] and d != D[-1]:
        sc_coef[(t,m-1,d,K[K.index(k)-1],c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    if m == M[-1] and k != K[0]:
        sc_coef[(t,m,d,K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if m == M[-1] and k != K[0] and d != D[0]:
        sc_coef[(t,m,D[D.index(d)-1],K[K.index(k)-1],c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and k != K[0] and d != D[-1]:
        sc_coef[(t,m,d,K[K.index(k)-1],c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    # part 5
    if (TL[c] <= m <= M[-2]) and k != K[-1]:
        sc_coef[(t,m-1,d,k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)]
    if (TL[c] <= m <= M[-2]) and k != K[-1] and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if (TL[c] <= m <= M[-2]) and k != K[-1] and d != D[-1]:
        sc_coef[(t,m-1,d,k,c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

    if m == M[-1] and k != K[-1]:
        sc_coef[(t,m-1,d,k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)]
    if m == M[-1] and k != K[-1] and d != D[0]:
        sc_coef[(t,m-1,D[D.index(d)-1],k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and k != K[-1] and d != D[-1]:
        sc_coef[(t,m-1,d,k,c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

    if m == M[-1] and k != K[-1]:
        sc_coef[(t,m,d,k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)]
    if m == M[-1] and k != K[-1] and d != D[0]:
        sc_coef[(t,m,D[D.index(d)-1],k,c)] += betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if m == M[-1] and k != K[-1] and d != D[-1]:
        sc_coef[(t,m,d,k,c)] -= betas['bpw'][(m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

# ps
for t,m,d,k,c in itertools.product(T,M,D,K,C):
    # part 1
    if (1 <= t <= T[-2]) and (1 <= m <= M[-2]):
        sc_coef[(t+1,m-1,d,k,c)] += betas['bps'][(t,m,d,k,c)]
    if (1 <= t <= T[-2]) and (m == M[-1]):
        sc_coef[(t+1,m-1,d,k,c)] += betas['bps'][(t,m,d,k,c)]
    if (1 <= t <= T[-2]) and (m == M[-1]):
        sc_coef[(t+1,m,d,k,c)] += betas['bps'][(t,m,d,k,c)]

    # part 2
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],k,c)] += betas['bps'][t,m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],k,c)] += betas['bps'][t,m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and d != D[0]:
        sc_coef[(t+1,m,D[D.index(d)-1],k,c)] += betas['bps'][t,m,d,k,c] * ptp_d[(D[D.index(d)-1],c)]
    
    # part 3
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and d != D[-1]:
        sc_coef[(t+1,m-1,d,k,c)] -= betas['bps'][t,m,d,k,c] * ptp_d[(d,c)]
    if (1 <= t <= T[-2]) and m == M[-1] and d != D[0]:
        sc_coef[(t+1,m-1,d,k,c)] -= betas['bps'][t,m,d,k,c] * ptp_d[(d,c)]
    if (1 <= t <= T[-2]) and m == M[-1] and d != D[0]:
        sc_coef[(t+1,m,d,k,c)] -= betas['bps'][t,m,d,k,c] * ptp_d[(d,c)]

    # part 4
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[0]:
        sc_coef[(t+1,m-1,d,K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[0] and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[0] and d != D[-1]:
        sc_coef[(t+1,m-1,d,K[K.index(k)-1],c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0]:
        sc_coef[(t+1,m-1,d,K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0] and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0] and d != D[-1]:
        sc_coef[(t+1,m-1,d,K[K.index(k)-1],c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0]:
        sc_coef[(t+1,m,d,K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0] and d != D[0]:
        sc_coef[(t+1,m,D[D.index(d)-1],K[K.index(k)-1],c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[0] and d != D[-1]:
        sc_coef[(t+1,m,d,K[K.index(k)-1],c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(K[K.index(k)-1],c)] * ptp_d[(d,c)]

    # part 5
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[-1]:
        sc_coef[(t+1,m-1,d,k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)]
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[-1] and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and (TL[c] <= m <= M[-2]) and k != K[-1] and d != D[-1]:
        sc_coef[(t+1,m-1,d,k,c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1]:
        sc_coef[(t+1,m-1,d,k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1] and d != D[0]:
        sc_coef[(t+1,m-1,D[D.index(d)-1],k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1] and d != D[-1]:
        sc_coef[(t+1,m-1,d,k,c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1]:
        sc_coef[(t+1,m,d,k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1] and d != D[0]:
        sc_coef[(t+1,m,D[D.index(d)-1],k,c)] -= betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(D[D.index(d)-1],c)]
    if (1 <= t <= T[-2]) and m == M[-1] and k != K[-1] and d != D[-1]:
        sc_coef[(t+1,m,d,k,c)] += betas['bps'][(t,m,d,k,c)] * ptp_k[(k,c)] * ptp_d[(d,c)]

# pprint(sc_coef)

#%%
coef_df = pd.Series(sc_coef).reset_index()
coef_df.columns = ['T','M','D','K','C','Val']
coef_df['DK'] = coef_df['D'] + " \t" + coef_df['K']

# for m in M:
#     fig = px.line(coef_df.query(f'M == {m}'), x='T',y='Val',color='C', facet_row='D', facet_col='K', title=f'Scheduling Objective - Wait List: {m}', markers=True)
#     fig.show(renderer="browser")


fig = px.line(coef_df, x='T',y='Val',color='DK', facet_row='C', facet_col='M', title=f'Scheduling Objective', markers=True)
fig.show(renderer="browser")
# %%

# %%
