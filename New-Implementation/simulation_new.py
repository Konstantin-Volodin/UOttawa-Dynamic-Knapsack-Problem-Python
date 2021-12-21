#%%
##### Initialization #####
#region
from os import linesep
from Modules import data_import

from gurobipy import *
import itertools
import os.path
#endregion

##### Read Data #####
#region
my_path = os.getcwd()
input_data = data_import.read_data(os.path.join(my_path, 'Data', 'simple-data.xlsx'))

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

# Betas
#endregion

#%%
##### Myopic Model #####
#region
myopic = Model('Myopic')
myopic.params.LogToConsole = 0


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
myc_uup = myopic.addConstrs((sv_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * sv_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
myc_pwp = myopic.addConstrs((sv_aux_pwp[(m,d,k,c)] == sv_st_pw[(m,d,k,c)] - quicksum( sv_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
myc_psp = myopic.addConstrs((sv_aux_psp[(t,m,d,k,c)] == sv_st_ps[(t,m,d,k,c)] + sv_ac_sc[(t,m,d,k,c)] + quicksum( sv_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( sv_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

myc_aux_uv_0 = myopic.addConstrs((sv_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
myc_aux_uv_0M = myopic.addConstrs((sv_aux_uv[(t,p)] <= BM * sv_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
myc_aux_uv_1 = myopic.addConstrs((sv_aux_uv[(1,p)] >= sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p] for p in P), name='con_auxiliary_uv_1')
myc_aux_uv_1M = myopic.addConstrs((sv_aux_uv[(1,p)] <= (sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p]) + BM * (1 - sv_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
myc_aux_uv_m = myopic.addConstrs((sv_aux_uv[(t, p)] >= (sv_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
myc_aux_uv_mM = myopic.addConstrs((sv_aux_uv[(t, p)] <= (sv_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - sv_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

myc_aux_ulp_0 = myopic.addConstrs((sv_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
myc_aux_ulp_0M = myopic.addConstrs((sv_aux_ulp[p] <= BM * sv_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
myc_aux_ulp_p = myopic.addConstrs((sv_aux_ulp[p] >= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
myc_aux_ulp_pM = myopic.addConstrs((sv_aux_ulp[p] <= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) + BM * (1-sv_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

myc_aux_pwt_d = myopic.addConstrs((sv_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_d')
myc_aux_pwt_k = myopic.addConstrs((sv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_k')
myc_aux_pst_d = myopic.addConstrs((sv_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * sv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_d')
myc_aux_pst_k = myopic.addConstrs((sv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * sv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_k')

# State Action Constraints
myc_usage_1 = myopic.addConstrs((sv_aux_uup[(1,p)] <= p_dat[p].expected_units + sv_st_ul[p] + sv_aux_uv[(1,p)] for p in P), name='con_usage_1')
myc_usage_tT = myopic.addConstrs((sv_aux_uup[(t,p)] <= p_dat[p].expected_units + sv_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

myc_rescb = myopic.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
myc_rescb_ttp = myopic.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

myc_resch = myopic.addConstrs((quicksum(sv_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= sv_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
myc_sched = myopic.addConstrs((quicksum(sv_ac_sc[(t,m,d,k,c)] for t in T) <= sv_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

myc_ul_bd = myopic.addConstrs((sv_st_ul[p] <= max(10, 2 * p_dat[p].expected_units) for p in P), name='con_ul_bound')
myc_pw_bd = myopic.addConstrs((sv_st_pw[(m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for m in M for d in D for k in K for c in C), name='con_pw_bound')
myc_ps_bd = myopic.addConstrs((sv_st_ps[(t,m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for t in T for m in M for d in D for k in K for c in C), name='con_pw_bound')
#endregion

##### MDP Model #####
#region
MDP = Model('MDP')
MDP.params.LogToConsole = 0


# State Action & Auxiliary Variables
myv_st_ul = MDP.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, name='var_state_ul')
myv_st_pw = MDP.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_pw')
myv_st_ps = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_ps')

myv_ac_sc = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
myv_ac_rsc = MDP.addVars(T, T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_rsc')

myv_aux_uv = MDP.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
myv_aux_uvb = MDP.addVars(T, P, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

myv_aux_ulp = MDP.addVars(PCO, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
myv_aux_ulb = MDP.addVars(PCO, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

myv_aux_uup = MDP.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
myv_aux_pwp = MDP.addVars(M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
myv_aux_psp = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

myv_aux_pwt_d = MDP.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
myv_aux_pwt_k = MDP.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
myv_aux_pst_d = MDP.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
myv_aux_pst_k = MDP.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

# Definition of auxiliary variables
myc_uup = MDP.addConstrs((sv_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * sv_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
myc_pwp = MDP.addConstrs((sv_aux_pwp[(m,d,k,c)] == sv_st_pw[(m,d,k,c)] - quicksum( sv_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
myc_psp = MDP.addConstrs((sv_aux_psp[(t,m,d,k,c)] == sv_st_ps[(t,m,d,k,c)] + sv_ac_sc[(t,m,d,k,c)] + quicksum( sv_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( sv_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

myc_aux_uv_0 = MDP.addConstrs((sv_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
myc_aux_uv_0M = MDP.addConstrs((sv_aux_uv[(t,p)] <= BM * sv_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
myc_aux_uv_1 = MDP.addConstrs((sv_aux_uv[(1,p)] >= sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p] for p in P), name='con_auxiliary_uv_1')
myc_aux_uv_1M = MDP.addConstrs((sv_aux_uv[(1,p)] <= (sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p]) + BM * (1 - sv_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
myc_aux_uv_m = MDP.addConstrs((sv_aux_uv[(t, p)] >= (sv_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
myc_aux_uv_mM = MDP.addConstrs((sv_aux_uv[(t, p)] <= (sv_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - sv_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

myc_aux_ulp_0 = MDP.addConstrs((sv_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
myc_aux_ulp_0M = MDP.addConstrs((sv_aux_ulp[p] <= BM * sv_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
myc_aux_ulp_p = MDP.addConstrs((sv_aux_ulp[p] >= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
myc_aux_ulp_pM = MDP.addConstrs((sv_aux_ulp[p] <= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) + BM * (1-sv_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

myc_aux_pwt_d = MDP.addConstrs((sv_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_d')
myc_aux_pwt_k = MDP.addConstrs((sv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_k')
myc_aux_pst_d = MDP.addConstrs((sv_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * sv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_d')
myc_aux_pst_k = MDP.addConstrs((sv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * sv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_k')

# State Action Constraints
myc_usage_1 = MDP.addConstrs((sv_aux_uup[(1,p)] <= p_dat[p].expected_units + sv_st_ul[p] + sv_aux_uv[(1,p)] for p in P), name='con_usage_1')
myc_usage_tT = MDP.addConstrs((sv_aux_uup[(t,p)] <= p_dat[p].expected_units + sv_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

myc_rescb = MDP.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
myc_rescb_ttp = MDP.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

myc_resch = MDP.addConstrs((quicksum(sv_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= sv_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
myc_sched = MDP.addConstrs((quicksum(sv_ac_sc[(t,m,d,k,c)] for t in T) <= sv_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

myc_ul_bd = MDP.addConstrs((sv_st_ul[p] <= max(10, 2 * p_dat[p].expected_units) for p in P), name='con_ul_bound')
myc_pw_bd = MDP.addConstrs((sv_st_pw[(m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for m in M for d in D for k in K for c in C), name='con_pw_bound')
myc_ps_bd = MDP.addConstrs((sv_st_ps[(t,m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for t in T for m in M for d in D for k in K for c in C), name='con_pw_bound')
