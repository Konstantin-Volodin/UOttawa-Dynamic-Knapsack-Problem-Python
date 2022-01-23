from os import close, linesep
from Modules import data_import

from gurobipy import *
import itertools
import os.path
import pickle
import time
from copy import deepcopy
import numpy as np
from tqdm.auto import trange


##### Parameters #####
#region
column_num = 1800
test_modifier = "cw1-cc5-cv100-gam95-"
data_type = "complex"
import_data = f"Data/sens-data/{data_type}/{test_modifier}{data_type}-data.xlsx"

# export_data_opt = f"Data/sens-data/{data_type}/betas/{test_modifier}{data_type}-optimal.pkl"
# export_data_p2 = f"Data/sens-data/{data_type}/full-model/{test_modifier}{data_type}-p2.mps"

#endregion


##### Read Data #####
#region
my_path = os.getcwd()
input_data = data_import.read_data(os.path.join(my_path, import_data))

# Quick Assess to Various Parameters
TL = input_data.transition.wait_limit
BM = 10000
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
state = {'ul': E_UL, 'pw': E_PW, 'ps': E_PS}
for i in itertools.product(M,D,K,C): state['pw'][i] = np.random.poisson(E_PW[i])
for i in itertools.product(T,M,D,K,C): state['ps'][i] = np.random.poisson(E_PS[i])
#endregion


##### Master Model #####
#region
master = Model("Master problem")
master.params.LogToConsole = 0
# master.params.Method = 1

# Goal Variables
mv_b0 = master.addVar(vtype = GRB.CONTINUOUS, lb=0, name='var_beta_0')

mv_bul_co = master.addVars(PCO, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ul_co')
mv_bul_nco = master.addVars(PNCO, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ul_nco')

mv_bpw_0 = master.addVars(M[0:1], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_0dkc')
mv_bpw_1TL = master.addVars(mTLdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_1TLdkc')
mv_bpw_TLM = master.addVars(TLMdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_TLMdkc')
mv_bpw_M = master.addVars(M[-1:], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_Mdkc')

mv_bps_0 = master.addVars(T[:-1], M[0:1], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_t0dkc')
mv_bps_1TL = master.addVars(tmTLdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_t1TLdkc')
mv_bps_TLM = master.addVars(tTLMdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_tTLMdkc')
mv_bps_M = master.addVars(T[:-1], M[-1:], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_tMdkc')
mv_bps_T = master.addVars(T[-1:], M, D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_Tmdkc')

# Constraint Definition
mc_b0 = master.addConstr(mv_b0 == 1, name='con_beta_0')

mc_bul_co = master.addConstrs((mv_bul_co[p] >= E_UL[p] for p in PCO), name='con_beta_ul_co')
mc_bul_nco = master.addConstrs((mv_bul_nco[p] >= E_UL[p] for p in PNCO), name='con_beta_ul_nco')

mc_bpw_0 = master.addConstrs((mv_bpw_0[(m, d, k, c)] >= E_PW[(m, d, k, c)] for m in M[0:1] for d in D for k in K for c in C), name='con_beta_pw_0dkc')
mc_bpw_1TL = master.addConstrs((mv_bpw_1TL[i] >= E_PW[i] for i in mTLdkc), name='con_beta_pw_1TLdkc')
mc_bpw_TLM = master.addConstrs((mv_bpw_TLM[i] >= E_PW[i] for i in TLMdkc), name='con_beta_pw_TLMdkc')
mc_bpw_M = master.addConstrs((mv_bpw_M[(m, d, k, c)] >= E_PW[(m, d, k, c)] for m in M[-1:] for d in D for k in K for c in C), name='con_beta_pw_Mdkc')

mc_bps_0 = master.addConstrs((mv_bps_0[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[:-1] for m in M[0:1] for d in D for k in K for c in C), name='con_beta_ps_t0dkc')
mc_bps_1TL = master.addConstrs((mv_bps_1TL[i] >= E_PS[i] for i in tmTLdkc), name='con_beta_ps_t1TLdkc')
mc_bps_TLM = master.addConstrs((mv_bps_TLM[i] >= E_PS[i] for i in tTLMdkc), name='con_beta_ps_tTLMdkc')
mc_bps_M = master.addConstrs((mv_bps_M[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[:-1] for m in M[-1:] for d in D for k in K for c in C ), name='con_beta_ps_tMdkc')
mc_bps_T = master.addConstrs((mv_bps_T[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[-1:] for m in M for d in D for k in K for c in C), name='con_beta_ps_Tmdkc')

# Objective Function Definition
mo_bul_co = quicksum(mv_bul_co[p] for p in PCO)
mo_bul_nco = quicksum(mv_bul_nco[p] for p in PNCO)

mo_bpw_0 = quicksum(mv_bpw_0[(0, d, k, c)] for d in D for k in K for c in C)
mo_bpw_1TL = quicksum(mv_bpw_1TL[i] for i in mTLdkc)
mo_bpw_TLM = quicksum(mv_bpw_TLM[i] for i in TLMdkc)
mo_bpw_M = quicksum(mv_bpw_M[(M[-1], d, k, c)] for d in D for k in K for c in C)

mo_bps_0 = quicksum(mv_bps_0[(t, 0, d, k, c)] for t in T[:-1] for d in D for k in K for c in C)
mo_bps_1TL = quicksum(mv_bps_1TL[i] for i in tmTLdkc)
mo_bps_TLM = quicksum(mv_bps_TLM[i] for i in tTLMdkc)
mo_bps_M = quicksum(mv_bps_M[t, M[-1], d, k, c] for t in T[:-1] for d in D for k in K for c in C)
mo_bps_T = quicksum(mv_bps_T[T[-1], m, d, k, c] for m in M for d in D for k in K for c in C)

master.setObjective( (mv_b0) + (mo_bul_co + mo_bul_nco) + (mo_bpw_0 + mo_bpw_1TL + mo_bpw_TLM + mo_bpw_M) + (mo_bps_0 + mo_bps_1TL + mo_bps_TLM + mo_bps_M + mo_bps_T), GRB.MINIMIZE)
#endregion


##### Sub Problem #####
#region
sub_prob = Model('Sub problem')
sub_prob.params.LogToConsole = 0
# sub_prob.params.DegenMoves = 2

# State Action & Auxiliary Variables
sv_st_ul = sub_prob.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, name='var_state_ul')
sv_st_pw = sub_prob.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_pw')
sv_st_ps = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_ps')

sv_ac_sc = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
sv_ac_rsc = sub_prob.addVars(T, T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_rsc')

sv_aux_uv = sub_prob.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
sv_aux_uvb = sub_prob.addVars(T, P, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

sv_aux_ulp = sub_prob.addVars(PCO, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
sv_aux_ulb = sub_prob.addVars(PCO, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

sv_aux_uup = sub_prob.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
sv_aux_pwp = sub_prob.addVars(M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
sv_aux_psp = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

sv_aux_pwt_d = sub_prob.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
sv_aux_pwt_k = sub_prob.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
sv_aux_pst_d = sub_prob.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
sv_aux_pst_k = sub_prob.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

# Definition of auxiliary variables
sc_uup = sub_prob.addConstrs((sv_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * sv_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
sc_pwp = sub_prob.addConstrs((sv_aux_pwp[(m,d,k,c)] == sv_st_pw[(m,d,k,c)] - quicksum( sv_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
sc_psp = sub_prob.addConstrs((sv_aux_psp[(t,m,d,k,c)] == sv_st_ps[(t,m,d,k,c)] + sv_ac_sc[(t,m,d,k,c)] + quicksum( sv_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( sv_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

sc_aux_uv_0 = sub_prob.addConstrs((sv_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
sc_aux_uv_0M = sub_prob.addConstrs((sv_aux_uv[(t,p)] <= BM * sv_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
sc_aux_uv_1 = sub_prob.addConstrs((sv_aux_uv[(1,p)] >= sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p] for p in P), name='con_auxiliary_uv_1')
sc_aux_uv_1M = sub_prob.addConstrs((sv_aux_uv[(1,p)] <= (sv_aux_uup[(1, p)] - p_dat[p].expected_units - sv_st_ul[p]) + BM * (1 - sv_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
sc_aux_uv_m = sub_prob.addConstrs((sv_aux_uv[(t, p)] >= (sv_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
sc_aux_uv_mM = sub_prob.addConstrs((sv_aux_uv[(t, p)] <= (sv_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - sv_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

sc_aux_ulp_0 = sub_prob.addConstrs((sv_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
sc_aux_ulp_0M = sub_prob.addConstrs((sv_aux_ulp[p] <= BM * sv_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
sc_aux_ulp_p = sub_prob.addConstrs((sv_aux_ulp[p] >= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
sc_aux_ulp_pM = sub_prob.addConstrs((sv_aux_ulp[p] <= (p_dat[p].expected_units + sv_st_ul[p] - sv_aux_uup[(1,p)]) + BM * (1-sv_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

sc_aux_pwt_d = sub_prob.addConstrs((sv_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwt_d')
sc_aux_pwt_k_0 = sub_prob.addConstrs((sv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_pwp[(m,d,k,c)] - sv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pwt_k_0')
sc_aux_pwt_k_i = sub_prob.addConstrs((sv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_pwp[(m,d,k,c)] + sv_aux_pwt_d[(m,D[D.index(d)-1],k,c)] - sv_aux_pwt_d[(m,d,k,c)]) for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pwt_k')
sc_aux_pwt_k_D = sub_prob.addConstrs((sv_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_pwp[(m,d,k,c)] + sv_aux_pwt_d[(m,D[D.index(d)-1],k,c)]) for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pwt_k_D')
sc_aux_pwt_k = {**sc_aux_pwt_k_0, **sc_aux_pwt_k_i, **sc_aux_pwt_k_D}

sc_aux_pst_d = sub_prob.addConstrs((sv_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * sv_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pst_d')
sc_aux_pst_k_0 = sub_prob.addConstrs((sv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_psp[(t,m,d,k,c)] - sv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[0]), name='con_auxiliary_pwt_k_0')
sc_aux_pst_k_i = sub_prob.addConstrs((sv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_psp[(t,m,d,k,c)] + sv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)] - sv_aux_pst_d[(t,m,d,k,c)]) for t in T for m in M for d in D for k in K for c in C if d != D[0] and d != D[-1]), name='con_auxiliary_pwt_k')
sc_aux_pst_k_D = sub_prob.addConstrs((sv_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * (sv_aux_psp[(t,m,d,k,c)] + sv_aux_pst_d[(t,m,D[D.index(d)-1],k,c)]) for t in T for m in M for d in D for k in K for c in C if d == D[-1]), name='con_auxiliary_pwt_k_D')
sc_aux_pst_k = {**sc_aux_pst_k_0, **sc_aux_pst_k_i, **sc_aux_pst_k_D}

# State Action Constraints
sc_usage_1 = sub_prob.addConstrs((sv_aux_uup[(1,p)] <= p_dat[p].expected_units + sv_st_ul[p] + sv_aux_uv[(1,p)] for p in P), name='con_usage_1')
sc_usage_tT = sub_prob.addConstrs((sv_aux_uup[(t,p)] <= p_dat[p].expected_units + sv_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

sc_rescb = sub_prob.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
sc_rescb_ttp = sub_prob.addConstrs((sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

sc_resch = sub_prob.addConstrs((quicksum(sv_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= sv_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
sc_sched = sub_prob.addConstrs((quicksum(sv_ac_sc[(t,m,d,k,c)] for t in T) <= sv_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

sc_ul_bd = sub_prob.addConstrs((sv_st_ul[p] <= max(10, 2 * p_dat[p].expected_units) for p in P), name='con_ul_bound')
sc_pw_bd = sub_prob.addConstrs((sv_st_pw[(m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for m in M for d in D for k in K for c in C), name='con_pw_bound')
sc_ps_bd = sub_prob.addConstrs((sv_st_ps[(t,m,d,k,c)] <= max(10, 5*pea[(d,k,c)]) for t in T for m in M for d in D for k in K for c in C), name='con_pw_bound')
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


##### Phase 1 #####
#region
mo_p2 = LinExpr()
iter = 0
while True:

    # Solve Master
    master.optimize()
    print(f"PHASE 1 Master Iter {iter}:\t\t{master.ObjVal}")


    # Generate Value Equations
    val_b0 = LinExpr(1-gam)

    val_bul_co = {p: LinExpr(sv_st_ul[p] - gam*sv_aux_ulp[p])  for p in PCO }
    val_bul_nco = {p: LinExpr(sv_st_ul[p])  for p in PNCO }

    val_bpw_0 = {(0,d,k,c): LinExpr(sv_st_pw[(0,d,k,c)] - gam*pea[(d,k,c)])  for d in D for k in K for c in C}
    val_bpw_1TL = {(m,d,k,c): LinExpr(sv_st_pw[(m,d,k,c)] - gam*sv_aux_pwp[(m-1,d,k,c)])  for m,d,k,c in mTLdkc}
    val_bpw_TLM = {}
    for m,d,k,c in TLMdkc:
        val_bpw_TLM[(m,d,k,c)] = LinExpr( sv_st_pw[(m,d,k,c)] - gam*sv_aux_pwp[(m-1,d,k,c)] )
        if d != D[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * sv_aux_pwt_d[(m-1, D[D.index(d)-1] ,k,c)]
        if k != K[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * sv_aux_pwt_k[(m-1,d, K[K.index(k)-1] ,c)]
        if d != D[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * sv_aux_pwt_d[(m-1,d,k,c)]
        if k != K[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * sv_aux_pwt_k[(m-1,d,k,c)]
    val_bpw_M = {}
    for m,d,k,c in itertools.product( M[-1:], D, K, C):
        val_bpw_M[(m,d,k,c)] = LinExpr( sv_st_pw[(m,d,k,c)] - gam*quicksum( sv_aux_pwp[mm,d,k,c] for mm in M[-2:] ) )
        if d != D[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( sv_aux_pwt_d[(mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
        if k != K[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( sv_aux_pwt_k[(mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
        if d != D[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( sv_aux_pwt_d[(mm,d,k,c)] for mm in M[-2:])
        if k != K[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( sv_aux_pwt_k[(mm,d,k,c)] for mm in M[-2:])    

    val_bps_0 = {(t,0,d,k,c): LinExpr(sv_st_ps[(t,0,d,k,c)])  for t in T[:-1] for d in D for k in K for c in C }
    val_bps_1TL = {(t,m,d,k,c): LinExpr(sv_st_ps[(t,m,d,k,c)] - gam*sv_aux_psp[(t+1,m-1,d,k,c)]) for t,m,d,k,c in tmTLdkc}
    val_bps_TLM = {}
    for t,m,d,k,c in tTLMdkc:
        val_bps_TLM[(t,m,d,k,c)] = LinExpr( sv_st_ps[(t,m,d,k,c)] - gam*sv_aux_psp[(t+1,m-1,d,k,c)] )
        if d != D[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * sv_aux_pst_d[(t+1,m-1, D[D.index(d)-1] ,k,c)]
        if k != K[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * sv_aux_pst_k[(t+1,m-1,d, K[K.index(k)-1] ,c)]
        if d != D[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * sv_aux_pst_d[(t+1,m-1,d,k,c)]
        if k != K[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * sv_aux_pst_k[(t+1,m-1,d,k,c)]
    val_bps_M = {}
    for t,m,d,k,c in itertools.product(T[:-1], M[-1:], D, K, C):
        val_bps_M[(t,m,d,k,c)] = LinExpr( sv_st_ps[(t,m,d,k,c)] - gam*quicksum( sv_aux_psp[(t+1,mm,d,k,c)] for mm in M[-2:] ) )
        if d != D[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( sv_aux_pst_d[(t+1,mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
        if k != K[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( sv_aux_pst_k[(t+1,mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
        if d != D[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( sv_aux_pst_d[(t+1,mm,d,k,c)] for mm in M[-2:])
        if k != K[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( sv_aux_pst_k[(t+1,mm,d,k,c)] for mm in M[-2:])    
    val_bps_T = {(T[-1],m,d,k,c): LinExpr( sv_st_ps[(T[-1],m,d,k,c)] )  for m in M for d in D for k in K for c in C}


    # Update Subproblem
    so_val =   ((   mc_b0.Pi * val_b0                                                       ) +
                (   quicksum(mc_bul_co[i].Pi*val_bul_co[i]      for i in mc_bul_co)     + 
                    quicksum(mc_bul_nco[i].Pi*val_bul_nco[i]    for i in mc_bul_nco)        ) + 
                (   quicksum(mc_bpw_0[i].Pi*val_bpw_0[i]        for i in mc_bpw_0)      + 
                    quicksum(mc_bpw_1TL[i].Pi*val_bpw_1TL[i]    for i in mc_bpw_1TL)    + 
                    quicksum(mc_bpw_TLM[i].Pi*val_bpw_TLM[i]    for i in mc_bpw_TLM)    +  
                    quicksum(mc_bpw_M[i].Pi*val_bpw_M[i]        for i in mc_bpw_M)          ) +
                (   quicksum(mc_bps_0[i].Pi*val_bps_0[i]        for i in mc_bps_0)      + 
                    quicksum(mc_bps_1TL[i].Pi*val_bps_1TL[i]    for i in mc_bps_1TL)    + 
                    quicksum(mc_bps_TLM[i].Pi*val_bps_TLM[i]    for i in mc_bps_TLM)    +  
                    quicksum(mc_bps_M[i].Pi*val_bps_M[i]        for i in mc_bps_M)      + 
                    quicksum(mc_bps_T[i].Pi*val_bps_T[i]        for i in mc_bps_T)          ))
    so_cw =     quicksum( cw[k] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    so_cs =     quicksum( cs[k][t] * sv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    so_brsc =   quicksum( (cs[k][tp-t]+cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    so_grsc =   quicksum( (cs[k][t-tp]-cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    so_cv =     quicksum( cv * sv_aux_uv[(t,p)] for  t in T for p in P)
    so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv)

    sub_prob.setObjective( -so_val, GRB.MINIMIZE )


    # Solve Subproblem
    sub_prob.optimize()


    # Update Master
    sa = Column()
    sa.addTerms(val_b0.getValue(),              mc_b0)
    
    [sa.addTerms(val_bul_co[i].getValue(),      mc_bul_co[i])   for i in mc_bul_co     ]    
    [sa.addTerms(val_bul_nco[i].getValue(),     mc_bul_nco[i])  for i in mc_bul_nco    ]

    [sa.addTerms(val_bpw_0[i].getValue(),       mc_bpw_0[i])    for i in mc_bpw_0      ]    
    [sa.addTerms(val_bpw_1TL[i].getValue(),     mc_bpw_1TL[i])  for i in mc_bpw_1TL    ]
    [sa.addTerms(val_bpw_TLM[i].getValue(),     mc_bpw_TLM[i])  for i in mc_bpw_TLM    ]    
    [sa.addTerms(val_bpw_M[i].getValue(),       mc_bpw_M[i])    for i in mc_bpw_M      ]

    [sa.addTerms(val_bps_0[i].getValue(),       mc_bps_0[i])    for i in mc_bps_0      ]    
    [sa.addTerms(val_bps_1TL[i].getValue(),     mc_bps_1TL[i])  for i in mc_bps_1TL    ]
    [sa.addTerms(val_bps_TLM[i].getValue(),     mc_bps_TLM[i])  for i in mc_bps_TLM    ]    
    [sa.addTerms(val_bps_M[i].getValue(),       mc_bps_M[i])    for i in mc_bps_M      ]
    [sa.addTerms(val_bps_T[i].getValue(),       mc_bps_T[i])    for i in mc_bps_T      ]  

    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa)


    # Save objective for phase 2
    mo_p2.add(sa_var, so_cost.getValue())


    # End Condition
    if master.ObjVal <= 0: 
        master.optimize()
        break

    iter += 1
#endregion    


##### Simulation
#region

# Prepare Master
master.remove(mv_b0)
master.remove(mv_bul_co)
master.remove(mv_bul_nco)
master.remove(mv_bpw_0)
master.remove(mv_bpw_1TL)
master.remove(mv_bpw_TLM)
master.remove(mv_bpw_M)
master.remove(mv_bps_0)
master.remove(mv_bps_1TL)
master.remove(mv_bps_TLM)
master.remove(mv_bps_M)
master.remove(mv_bps_T)
master.setObjective(mo_p2, GRB.MINIMIZE)

# Single Replication
for col in range(column_num):

    # Reset State 
    if col%300 == 0:
        state = {'ul': E_UL, 'pw': E_PW, 'ps': E_PS}
        # for i in itertools.product(M,D,K,C): state['pw'][i] = np.random.poisson(E_PW[i])
        # for i in itertools.product(T,M,D,K,C): state['ps'][i] = np.random.poisson(E_PS[i])
        for i in itertools.product(M,D,K,C): state['pw'][i] = 50
        for i in itertools.product(T,M,D,K,C): state['ps'][i] = 10

    # Different Policies to save columns from
    if col >= column_num/2:
        for k in K: myv_cost_cw[k].UB = cv-1; myv_cost_cw[k].LB = cv-1;
    else:
        for k in K: myv_cost_cw[k].UB = cw[k]; myv_cost_cw[k].LB = cw[k];


    # Evaluate Performance
    if col%50 == 0:

        # Get Betas
        beta = {'b0': mc_b0.Pi, 'bul': {}, 'bpw': {}, 'bps': {}}

        for p in PCO: beta['bul'][p] = mc_bul_co[p].Pi 
        for p in PNCO: beta['bul'][p] = mc_bul_nco[p].Pi 

        for i in itertools.product(M[0:1], D, K, C): beta['bpw'][i] = mc_bpw_0[i].Pi
        for i in mTLdkc: beta['bpw'][i] = mc_bpw_1TL[i].Pi
        for i in TLMdkc: beta['bpw'][i] = mc_bpw_TLM[i].Pi
        for i in itertools.product(M[-1:], D, K, C): beta['bpw'][i] = mc_bpw_M[i].Pi

        for i in itertools.product(T[:-1], M[0:1], D, K, C): beta['bps'][i] = mc_bps_0[i].Pi
        for i in tmTLdkc: beta['bps'][i] = mc_bps_1TL[i].Pi
        for i in tTLMdkc: beta['bps'][i] = mc_bps_TLM[i].Pi
        for i in itertools.product(T[:-1], M[-1:], D, K, C): beta['bps'][i] = mc_bps_M[i].Pi
        for i in itertools.product(T[-1:], M, D, K, C): beta['bps'][i] = mc_bps_T[i].Pi

        # Generate Column
        val_b0 = LinExpr(1-gam)

        val_bul_co = {p: LinExpr(sv_st_ul[p] - gam*sv_aux_ulp[p])  for p in PCO }
        val_bul_nco = {p: LinExpr(sv_st_ul[p])  for p in PNCO }

        val_bpw_0 = {(0,d,k,c): LinExpr(sv_st_pw[(0,d,k,c)] - gam*pea[(d,k,c)])  for d in D for k in K for c in C}
        val_bpw_1TL = {(m,d,k,c): LinExpr(sv_st_pw[(m,d,k,c)] - gam*sv_aux_pwp[(m-1,d,k,c)])  for m,d,k,c in mTLdkc}
        val_bpw_TLM = {}
        for m,d,k,c in TLMdkc:
            val_bpw_TLM[(m,d,k,c)] = LinExpr( sv_st_pw[(m,d,k,c)] - gam*sv_aux_pwp[(m-1,d,k,c)] )
            if d != D[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * sv_aux_pwt_d[(m-1, D[D.index(d)-1] ,k,c)]
            if k != K[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * sv_aux_pwt_k[(m-1,d, K[K.index(k)-1] ,c)]
            if d != D[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * sv_aux_pwt_d[(m-1,d,k,c)]
            if k != K[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * sv_aux_pwt_k[(m-1,d,k,c)]
        val_bpw_M = {}
        for m,d,k,c in itertools.product( M[-1:], D, K, C):
            val_bpw_M[(m,d,k,c)] = LinExpr( sv_st_pw[(m,d,k,c)] - gam*quicksum( sv_aux_pwp[mm,d,k,c] for mm in M[-2:] ) )
            if d != D[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( sv_aux_pwt_d[(mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
            if k != K[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( sv_aux_pwt_k[(mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
            if d != D[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( sv_aux_pwt_d[(mm,d,k,c)] for mm in M[-2:])
            if k != K[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( sv_aux_pwt_k[(mm,d,k,c)] for mm in M[-2:])    

        val_bps_0 = {(t,0,d,k,c): LinExpr(sv_st_ps[(t,0,d,k,c)])  for t in T[:-1] for d in D for k in K for c in C }
        val_bps_1TL = {(t,m,d,k,c): LinExpr(sv_st_ps[(t,m,d,k,c)] - gam*sv_aux_psp[(t+1,m-1,d,k,c)]) for t,m,d,k,c in tmTLdkc}
        val_bps_TLM = {}
        for t,m,d,k,c in tTLMdkc:
            val_bps_TLM[(t,m,d,k,c)] = LinExpr( sv_st_ps[(t,m,d,k,c)] - gam*sv_aux_psp[(t+1,m-1,d,k,c)] )
            if d != D[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * sv_aux_pst_d[(t+1,m-1, D[D.index(d)-1] ,k,c)]
            if k != K[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * sv_aux_pst_k[(t+1,m-1,d, K[K.index(k)-1] ,c)]
            if d != D[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * sv_aux_pst_d[(t+1,m-1,d,k,c)]
            if k != K[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * sv_aux_pst_k[(t+1,m-1,d,k,c)]
        val_bps_M = {}
        for t,m,d,k,c in itertools.product(T[:-1], M[-1:], D, K, C):
            val_bps_M[(t,m,d,k,c)] = LinExpr( sv_st_ps[(t,m,d,k,c)] - gam*quicksum( sv_aux_psp[(t+1,mm,d,k,c)] for mm in M[-2:] ) )
            if d != D[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( sv_aux_pst_d[(t+1,mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
            if k != K[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( sv_aux_pst_k[(t+1,mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
            if d != D[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( sv_aux_pst_d[(t+1,mm,d,k,c)] for mm in M[-2:])
            if k != K[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( sv_aux_pst_k[(t+1,mm,d,k,c)] for mm in M[-2:])    
        val_bps_T = {(T[-1],m,d,k,c): LinExpr( sv_st_ps[(T[-1],m,d,k,c)] )  for m in M for d in D for k in K for c in C}


        # Update Subproblem
        so_val =   ((   beta['b0'] * val_b0                                                       ) +
                    (   quicksum(beta['bul'][i]*val_bul_co[i]        for i in mc_bul_co)     + 
                        quicksum(beta['bul'][i]*val_bul_nco[i]       for i in mc_bul_nco)        ) + 
                    (   quicksum(beta['bpw'][i]*val_bpw_0[i]         for i in mc_bpw_0)      + 
                        quicksum(beta['bpw'][i]*val_bpw_1TL[i]       for i in mc_bpw_1TL)    + 
                        quicksum(beta['bpw'][i]*val_bpw_TLM[i]       for i in mc_bpw_TLM)    +  
                        quicksum(beta['bpw'][i]*val_bpw_M[i]         for i in mc_bpw_M)          ) +
                    (   quicksum(beta['bps'][i]*val_bps_0[i]         for i in mc_bps_0)      + 
                        quicksum(beta['bps'][i]*val_bps_1TL[i]       for i in mc_bps_1TL)    + 
                        quicksum(beta['bps'][i]*val_bps_TLM[i]       for i in mc_bps_TLM)    +  
                        quicksum(beta['bps'][i]*val_bps_M[i]         for i in mc_bps_M)      + 
                        quicksum(beta['bps'][i]*val_bps_T[i]         for i in mc_bps_T)          ))
        so_cw =     quicksum( cw[k] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
        so_cs =     quicksum( cs[k][t] * sv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
        so_brsc =   quicksum( (cs[k][tp-t]+cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
        so_grsc =   quicksum( (cs[k][t-tp]-cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
        so_cv =     quicksum( cv * sv_aux_uv[(t,p)] for  t in T for p in P)
        so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv)

        sub_prob.setObjective( so_cost-so_val, GRB.MINIMIZE )
        master.optimize()
        sub_prob.optimize()
        print(f"SIMULATION-STARTUP Col {col}:\t\tSUB: {sub_prob.ObjVal}\t\tMAST:{master.ObjVal}")

    # Generate Action (With new logic)
    for p in P: myv_st_ul[p].UB = state['ul'][p]; myv_st_ul[p].LB = state['ul'][p]
    for m,d,k,c in itertools.product(M, D, K, C): myv_st_pw[(m,d,k,c)].UB = round(state['pw'][(m,d,k,c)],0); myv_st_pw[(m,d,k,c)].LB = round(state['pw'][(m,d,k,c)],0)
    for t,m,d,k,c in itertools.product(T, M, D, K, C): myv_st_ps[(t,m,d,k,c)].UB = round(state['ps'][(t,m,d,k,c)],0); myv_st_ps[(t,m,d,k,c)].LB = round(state['ps'][(t,m,d,k,c)],0)
    myopic.optimize()

    # Save Action
    action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
    for i in itertools.product(T,M,D,K,C): action['sc'][i] = myv_ac_sc[i].X 
    for i in itertools.product(T,T,M,D,K,C): action['rsc'][i] = myv_ac_rsc[i].X
    for i in itertools.product(T,P): action['uv'][i] = myv_aux_uv[i].X
    for i in PCO: action['ulp'][i] = myv_aux_ulp[i].X
    for i in itertools.product(T,P): action['uup'][i] = myv_aux_uup[i].X
    for i in itertools.product(M,D,K,C): action['pwp'][i] = myv_aux_pwp[i].X
    for i in itertools.product(T,M,D,K,C): action['psp'][i] = myv_aux_psp[i].X

    # Generates Expressions for state-action coefficients
    val_b0 = LinExpr(1-gam)

    val_bul_co = {p: LinExpr(myv_st_ul[p] - gam*myv_aux_ulp[p])  for p in PCO }
    val_bul_nco = {p: LinExpr(myv_st_ul[p])  for p in PNCO }

    val_bpw_0 = {(0,d,k,c): LinExpr(myv_st_pw[(0,d,k,c)] - gam*pea[(d,k,c)])  for d in D for k in K for c in C}
    val_bpw_1TL = {(m,d,k,c): LinExpr(myv_st_pw[(m,d,k,c)] - gam*myv_aux_pwp[(m-1,d,k,c)])  for m,d,k,c in mTLdkc}
    val_bpw_TLM = {}
    for m,d,k,c in TLMdkc:
        val_bpw_TLM[(m,d,k,c)] = LinExpr( myv_st_pw[(m,d,k,c)] - gam*myv_aux_pwp[(m-1,d,k,c)] )
        if d != D[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * myv_aux_pwt_d[(m-1, D[D.index(d)-1] ,k,c)]
        if k != K[0]:  val_bpw_TLM[(m,d,k,c)] += -gam * myv_aux_pwt_k[(m-1,d, K[K.index(k)-1] ,c)]
        if d != D[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * myv_aux_pwt_d[(m-1,d,k,c)]
        if k != K[-1]: val_bpw_TLM[(m,d,k,c)] +=  gam * myv_aux_pwt_k[(m-1,d,k,c)]
    val_bpw_M = {}
    for m,d,k,c in itertools.product( M[-1:], D, K, C):
        val_bpw_M[(m,d,k,c)] = LinExpr( myv_st_pw[(m,d,k,c)] - gam*quicksum( myv_aux_pwp[mm,d,k,c] for mm in M[-2:] ) )
        if d != D[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( myv_aux_pwt_d[(mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
        if k != K[0]:  val_bpw_M[(m,d,k,c)] += -gam * quicksum( myv_aux_pwt_k[(mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
        if d != D[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( myv_aux_pwt_d[(mm,d,k,c)] for mm in M[-2:])
        if k != K[-1]: val_bpw_M[(m,d,k,c)] +=  gam * quicksum( myv_aux_pwt_k[(mm,d,k,c)] for mm in M[-2:])    

    val_bps_0 = {(t,0,d,k,c): LinExpr(myv_st_ps[(t,0,d,k,c)])  for t in T[:-1] for d in D for k in K for c in C }
    val_bps_1TL = {(t,m,d,k,c): LinExpr(myv_st_ps[(t,m,d,k,c)] - gam*myv_aux_psp[(t+1,m-1,d,k,c)]) for t,m,d,k,c in tmTLdkc}
    val_bps_TLM = {}
    for t,m,d,k,c in tTLMdkc:
        val_bps_TLM[(t,m,d,k,c)] = LinExpr( myv_st_ps[(t,m,d,k,c)] - gam*myv_aux_psp[(t+1,m-1,d,k,c)] )
        if d != D[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * myv_aux_pst_d[(t+1,m-1, D[D.index(d)-1] ,k,c)]
        if k != K[0]:  val_bps_TLM[(t,m,d,k,c)] += -gam * myv_aux_pst_k[(t+1,m-1,d, K[K.index(k)-1] ,c)]
        if d != D[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * myv_aux_pst_d[(t+1,m-1,d,k,c)]
        if k != K[-1]: val_bps_TLM[(t,m,d,k,c)] +=  gam * myv_aux_pst_k[(t+1,m-1,d,k,c)]
    val_bps_M = {}
    for t,m,d,k,c in itertools.product(T[:-1], M[-1:], D, K, C):
        val_bps_M[(t,m,d,k,c)] = LinExpr( myv_st_ps[(t,m,d,k,c)] - gam*quicksum( myv_aux_psp[(t+1,mm,d,k,c)] for mm in M[-2:] ) )
        if d != D[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( myv_aux_pst_d[(t+1,mm, D[D.index(d)-1] ,k,c)] for mm in M[-2:])
        if k != K[0]:  val_bps_M[(t,m,d,k,c)] += -gam * quicksum( myv_aux_pst_k[(t+1,mm,d, K[K.index(k)-1] ,c)] for mm in M[-2:])
        if d != D[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( myv_aux_pst_d[(t+1,mm,d,k,c)] for mm in M[-2:])
        if k != K[-1]: val_bps_M[(t,m,d,k,c)] +=  gam * quicksum( myv_aux_pst_k[(t+1,mm,d,k,c)] for mm in M[-2:])    
    val_bps_T = {(T[-1],m,d,k,c): LinExpr( myv_st_ps[(T[-1],m,d,k,c)] )  for m in M for d in D for k in K for c in C}
    so_cw =     quicksum( cw[k] * sv_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    so_cs =     quicksum( cs[k][t] * sv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    so_brsc =   quicksum( (cs[k][tp-t]+cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    so_grsc =   quicksum( (cs[k][t-tp]-cc[k]) * sv_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    so_cv =     quicksum( cv * sv_aux_uv[(t,p)] for  t in T for p in P)
    so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv)

    # Update Master
    sa = Column()
    sa.addTerms(val_b0.getValue(),              mc_b0)
    
    [sa.addTerms(val_bul_co[i].getValue(),      mc_bul_co[i])   for i in mc_bul_co     ]    
    [sa.addTerms(val_bul_nco[i].getValue(),     mc_bul_nco[i])  for i in mc_bul_nco    ]

    [sa.addTerms(val_bpw_0[i].getValue(),       mc_bpw_0[i])    for i in mc_bpw_0      ]    
    [sa.addTerms(val_bpw_1TL[i].getValue(),     mc_bpw_1TL[i])  for i in mc_bpw_1TL    ]
    [sa.addTerms(val_bpw_TLM[i].getValue(),     mc_bpw_TLM[i])  for i in mc_bpw_TLM    ]    
    [sa.addTerms(val_bpw_M[i].getValue(),       mc_bpw_M[i])    for i in mc_bpw_M      ]

    [sa.addTerms(val_bps_0[i].getValue(),       mc_bps_0[i])    for i in mc_bps_0      ]    
    [sa.addTerms(val_bps_1TL[i].getValue(),     mc_bps_1TL[i])  for i in mc_bps_1TL    ]
    [sa.addTerms(val_bps_TLM[i].getValue(),     mc_bps_TLM[i])  for i in mc_bps_TLM    ]    
    [sa.addTerms(val_bps_M[i].getValue(),       mc_bps_M[i])    for i in mc_bps_M      ]
    [sa.addTerms(val_bps_T[i].getValue(),       mc_bps_T[i])    for i in mc_bps_T      ]  

    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa, obj=so_cost.getValue())

    # Monitor Improvements
    

    # Transition between States
    # Units Leftover / Unit Deviation
    for p in PCO: state['ul'][p] = myv_aux_ulp[p].X + round(np.random.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)
    for p in PNCO: state['ul'][p] = round(np.random.uniform(p_dat[p].deviation[0],p_dat[p].deviation[1]), 2)

    # Patients Waiting - set to post action
    for m,d,k,c in itertools.product(M, D, K, C): state['pw'][(m,d,k,c)] = myv_aux_pwp[(m,d,k,c)].X
    # Patients Waiting - calculate & execute D Transition
    my_ptw_d = {}
    for m,d,k,c in itertools.product(M, D, K, C):
        if d != D[-1] and m >= TL[c]: my_ptw_d[(m,d,k,c)] = np.random.binomial(state['pw'][(m,d,k,c)], ptp_d[(d,c)] )
    for m,d,k,c in itertools.product(M, D, K, C): 
        if d != D[0] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] + my_ptw_d[(m,D[D.index(d)-1],k,c)]
    # Patients Waiting - calculate & execute K Transition
    my_ptw_k = {}
    for m,d,k,c in itertools.product(M, D, K, C):
        if k != K[-1] and m >= TL[c]: my_ptw_k[(m,d,k,c)] = np.random.binomial(state['pw'][(m,d,k,c)], ptp_k[(k,c)] )
    for m,d,k,c in itertools.product(M, D, K, C): 
        if d != D[-1] and m >= TL[c]: state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_d[(m,d,k,c)]      
    # Patients Waiting - change wait time
    for d,k,c in itertools.product(D, K, C): state['pw'][(M[-1],d,k,c)] +=  state['pw'][(M[-2],d,k,c)]
    for m,d,k,c in itertools.product(M[1:-1][::-1], D, K, C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]
    for d,k,c in itertools.product(D, K, C): state['pw'][(0,d,k,c)] = np.random.poisson(pea[(d,k,c)])

    # Patients Scheduled - post action
    for t,m,d,k,c in itertools.product(T, M, D, K, C): state['ps'][(t,m,d,k,c)] = myv_aux_psp[(t,m,d,k,c)].X
    # Patients Scheduled - calculate & execute D Transition
    my_pts_d = {}
    for t,m,d,k,c in itertools.product(T, M, D, K, C):
        if d != D[-1] and m >= TL[c]: my_pts_d[(t,m,d,k,c)] = np.random.binomial(state['ps'][(t,m,d,k,c)], ptp_d[(d,c)] )
    for t,m,d,k,c in itertools.product(T, M, D, K, C): 
        if d != D[0] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] + my_pts_d[(t,m,D[D.index(d)-1],k,c)]
    # Patients Scheduled - calculate & execute K Transition
    my_pts_k = {}
    for t,m,d,k,c in itertools.product(T, M, D, K, C):
        if k != K[-1] and m >= TL[c]: my_pts_k[(t,m,d,k,c)] = np.random.binomial(state['ps'][(t,m,d,k,c)], ptp_k[(k,c)] )
    for t,m,d,k,c in itertools.product(T, M, D, K, C): 
        if d != D[-1] and m >= TL[c]: state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_d[(t,m,d,k,c)]     
    # Patients Scheduled  - change wait time
    for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,M[-1],d,k,c)] +=  state['ps'][(t,M[-2],d,k,c)]
    for t,m,d,k,c in itertools.product(T, M[1:-1][::-1], D, K, C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
    for t,d,k,c in itertools.product(T, D, K, C): state['ps'][(t,0,d,k,c)] = 0
    # Patients Scheduled  - change scheduled time
    for t,m,d,k,c in itertools.product(T[:-1],M,D,K,C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
    for m,d,k,c in itertools.product(M,D,K,C): state['ps'][(T[-1],m,d,k,c)] = 0


#endregion


