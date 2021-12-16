#%%
##### Initialization #####
#region
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
#endregion

#%%
##### Master Phase 1 #####
#region
master = Model("Master problem")
master.params.LogToConsole = 0

# Goal Variables
ma_var_b0 = master.addVar(vtype = GRB.CONTINUOUS, lb=0, name='var_beta_0')

ma_var_bul_co = master.addVars(PCO, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ul_co')
ma_var_bul_nco = master.addVars(PNCO, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ul_nco')

ma_var_bpw_0 = master.addVars(M[0:1], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_0dkc')
ma_var_bpw_1TL = master.addVars(mTLdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_1TLdkc')
ma_var_bpw_TLM = master.addVars(TLMdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_TLMdkc')
ma_var_bpw_M = master.addVars(M[-1:], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_pw_Mdkc')

ma_var_bps_0 = master.addVars(T[:-1], M[0:1], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_t0dkc')
ma_var_bps_1TL = master.addVars(tmTLdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_t1TLdkc')
ma_var_bps_TLM = master.addVars(tTLMdkc, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_tTLMdkc')
ma_var_bps_M = master.addVars(T[:-1], M[-1:], D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_tMdkc')
ma_var_bps_T = master.addVars(T[-1:], M, D, K, C, vtype = GRB.CONTINUOUS, lb=0, name='var_beta_ps_Tmdkc')

# Constraint Definition
ma_con_b0 = master.addConstr(ma_var_b0 == 1, name='con_beta_0')

ma_con_bul_co = master.addConstrs((ma_var_bul_co[p] >= E_UL[p] for p in PCO), name='con_beta_ul_co')
ma_con_bul_nco = master.addConstrs((ma_var_bul_nco[p] >= E_UL[p] for p in PNCO), name='con_beta_ul_nco')

ma_con_bpw_0 = master.addConstrs((ma_var_bpw_0[(m, d, k, c)] >= E_PW[(m, d, k, c)] for m in M[0:1] for d in D for k in K for c in C), name='con_beta_pw_0dkc')
ma_con_bpw_1TL = master.addConstrs((ma_var_bpw_1TL[i] >= E_PW[i] for i in mTLdkc), name='con_beta_pw_1TLdkc')
ma_con_bpw_TLM = master.addConstrs((ma_var_bpw_TLM[i] >= E_PW[i] for i in TLMdkc), name='con_beta_pw_TLMdkc')
ma_con_bpw_M = master.addConstrs((ma_var_bpw_M[(m, d, k, c)] >= E_PW[(m, d, k, c)] for m in M[-1:] for d in D for k in K for c in C), name='con_beta_pw_Mdkc')

ma_con_bps_0 = master.addConstrs((ma_var_bps_0[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[:-1] for m in M[0:1] for d in D for k in K for c in C), name='con_beta_ps_t0dkc')
ma_con_bps_1TL = master.addConstrs((ma_var_bps_1TL[i] >= E_PS[i] for i in tmTLdkc), name='con_beta_ps_t1TLdkc')
ma_con_bps_TLM = master.addConstrs((ma_var_bps_TLM[i] >= E_PS[i] for i in tTLMdkc), name='con_beta_ps_tTLMdkc')
ma_con_bps_M = master.addConstrs((ma_var_bps_M[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[:-1] for m in M[-1:] for d in D for k in K for c in C ), name='con_beta_ps_tMdkc')
ma_con_bps_T = master.addConstrs((ma_var_bps_T[(t, m, d, k, c)] >= E_PS[(t, m, d, k, c)] for t in T[-1:] for m in M for d in D for k in K for c in C), name='con_beta_ps_Tmdkc')

# Objective Function Definition
ma_obj_bul_co = quicksum(ma_var_bul_co[p] for p in PCO)
ma_obj_bul_nco = quicksum(ma_var_bul_nco[p] for p in PNCO)

ma_obj_bpw_0 = quicksum(ma_var_bpw_0[(0, d, k, c)] for d in D for k in K for c in C)
ma_obj_bpw_1TL = quicksum(ma_var_bpw_1TL[i] for i in mTLdkc)
ma_obj_bpw_TLM = quicksum(ma_var_bpw_TLM[i] for i in TLMdkc)
ma_obj_bpw_M = quicksum(ma_var_bpw_M[(M[-1], d, k, c)] for d in D for k in K for c in C)

ma_obj_bps_0 = quicksum(ma_var_bps_0[(t, 0, d, k, c)] for t in T[:-1] for d in D for k in K for c in C)
ma_obj_bps_1TL = quicksum(ma_var_bps_1TL[i] for i in tmTLdkc)
ma_obj_bps_TLM = quicksum(ma_var_bps_TLM[i] for i in tTLMdkc)
ma_obj_bps_M = quicksum(ma_var_bps_M[t, M[-1], d, k, c] for t in T[:-1] for d in D for k in K for c in C)
ma_obj_bps_T = quicksum(ma_var_bps_T[T[-1], m, d, k, c] for m in M for d in D for k in K for c in C)

master.setObjective( (ma_var_b0) + (ma_obj_bul_co + ma_obj_bul_nco) + (ma_obj_bpw_0 + ma_obj_bpw_1TL + ma_obj_bpw_TLM + ma_obj_bpw_M) + (ma_obj_bps_0 + ma_obj_bps_1TL + ma_obj_bps_TLM + ma_obj_bps_M + ma_obj_bps_T), GRB.MINIMIZE)
#endregion

#%%
##### Sub Problem #####
#region
sub_prob = Model('Sub problem')
sub_prob.params.LogToConsole = 0

# State Action & Auxiliary Variables
sub_var_st_ul = sub_prob.addVars(P, vtype=GRB.CONTINUOUS, lb = 0, name='var_state_ul')
sub_var_st_pw = sub_prob.addVars(M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_pw')
sub_var_st_ps = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb=0, name='var_state_ps')

sub_var_ac_sc = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
sub_var_ac_rsc = sub_prob.addVars(T, T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_rsc')

sub_var_aux_uv = sub_prob.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
sub_var_aux_uvb = sub_prob.addVars(T, P, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

sub_var_aux_ulp = sub_prob.addVars(PCO, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
sub_var_aux_ulb = sub_prob.addVars(PCO, vtype=GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

sub_var_aux_uup = sub_prob.addVars(T, P, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
sub_var_aux_pwp = sub_prob.addVars(M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
sub_var_aux_psp = sub_prob.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

sub_var_aux_pwt_d = sub_prob.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
sub_var_aux_pwt_k = sub_prob.addVars(M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
sub_var_aux_pst_d = sub_prob.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
sub_var_aux_pst_k = sub_prob.addVars(T,M,D,K,C, vtype=GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

# Definition of auxiliary variables
sub_con_uup = sub_prob.addConstrs((sub_var_aux_uup[(t,p)] == quicksum( U[(p,d,c)] * sub_var_aux_psp[(t,m,d,k,c)] for m in M for d in D for k in K for c in C) for t in T for p in P), name='con_auxiliary_uup')
sub_con_pwp = sub_prob.addConstrs((sub_var_aux_pwp[(m,d,k,c)] == sub_var_st_pw[(m,d,k,c)] - quicksum( sub_var_ac_sc[(t,m,d,k,c)] for t in T) for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp')
sub_con_psp = sub_prob.addConstrs((sub_var_aux_psp[(t,m,d,k,c)] == sub_var_st_ps[(t,m,d,k,c)] + sub_var_ac_sc[(t,m,d,k,c)] + quicksum( sub_var_ac_rsc[tp,t,m,d,k,c] for tp in T) - quicksum( sub_var_ac_rsc[t,tp,m,d,k,c] for tp in T) for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_pws')

sub_con_aux_uv_0 = sub_prob.addConstrs((sub_var_aux_uv[(t,p)] >= 0 for t in T for p in P), name='con_auxiliary_uv_0')
sub_con_aux_uv_0M = sub_prob.addConstrs((sub_var_aux_uv[(t,p)] <= BM * sub_var_aux_uvb[(t,p)] for t in T for p in P), name='con_auxiliary_uv_0M')
sub_con_aux_uv_1 = sub_prob.addConstrs((sub_var_aux_uv[(1,p)] >= sub_var_aux_uup[(1, p)] - p_dat[p].expected_units - sub_var_st_ul[p] for p in P), name='con_auxiliary_uv_1')
sub_con_aux_uv_1M = sub_prob.addConstrs((sub_var_aux_uv[(1,p)] <= (sub_var_aux_uup[(1, p)] - p_dat[p].expected_units - sub_var_st_ul[p]) + BM * (1 - sub_var_aux_uvb[(1, p)]) for p in P), name='con_auxiliary_uv_1M')
sub_con_aux_uv_m = sub_prob.addConstrs((sub_var_aux_uv[(t, p)] >= (sub_var_aux_uup[(t, p)] - p_dat[p].expected_units) for t in T[1:] for p in P), name='con_auxiliary_uv_m')
sub_con_aux_uv_mM = sub_prob.addConstrs((sub_var_aux_uv[(t, p)] <= (sub_var_aux_uup[(t,p)] - p_dat[p].expected_units) + BM * (1 - sub_var_aux_uvb[(t, p)]) for t in T[1:] for p in P), name='con_auxiliary_uv_mM')

sub_con_aux_ulp_0 = sub_prob.addConstrs((sub_var_aux_ulp[p] >= 0 for p in PCO), name='con_auxiliary_ulp_0')
sub_con_aux_ulp_0M = sub_prob.addConstrs((sub_var_aux_ulp[p] <= BM * sub_var_aux_ulb[p] for p in PCO), name='con_auxiliary_ulp_0M')
sub_con_aux_ulp_p = sub_prob.addConstrs((sub_var_aux_ulp[p] >= (p_dat[p].expected_units + sub_var_st_ul[p] - sub_var_aux_uup[(1,p)]) for p in PCO), name='con_auxiliary_ulp_p')
sub_con_aux_ulp_pM = sub_prob.addConstrs((sub_var_aux_ulp[p] <= (p_dat[p].expected_units + sub_var_st_ul[p] - sub_var_aux_uup[(1,p)]) + BM * (1-sub_var_aux_ulb[p]) for p in PCO), name='con_auxiliary_ulp_pM')

sub_con_aux_pwt_d = sub_prob.addConstrs((sub_var_aux_pwt_d[(m,d,k,c)] == ptp_d[(d,c)] * sub_var_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_d')
sub_con_aux_pwt_k = sub_prob.addConstrs((sub_var_aux_pwt_k[(m,d,k,c)] == ptp_k[(k,c)] * sub_var_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_auxiliary_pwp_k')
sub_con_aux_pst_d = sub_prob.addConstrs((sub_var_aux_pst_d[(t,m,d,k,c)] == ptp_d[(d,c)] * sub_var_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_d')
sub_con_aux_pst_k = sub_prob.addConstrs((sub_var_aux_pst_k[(t,m,d,k,c)] == ptp_k[(k,c)] * sub_var_aux_psp[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_auxiliary_psp_k')

# State Action Constraints
sub_con_usage_1 = sub_prob.addConstrs((sub_var_aux_uup[(1,p)] <= p_dat[p].expected_units + sub_var_st_ul[p] + sub_var_aux_uv[(1,p)] for p in P), name='con_usage_1')
sub_con_usage_tT = sub_prob.addConstrs((sub_var_aux_uup[(t,p)] <= p_dat[p].expected_units + sub_var_aux_uv[(t,p)] for t in T[1:] for p in P), name='con_usage_tT')

sub_con_rescb = sub_prob.addConstrs((sub_var_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T[1:] for tp in T[1:] for m in M for d in D for k in K for c in C), name='con_reschedule_bounds')
sub_con_rescb_ttp = sub_prob.addConstrs((sub_var_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in T for tp in T for m in M for d in D for k in K for c in C if t == tp == 1), name='con_reschedule_bounds_ttp')

sub_con_resch = sub_prob.addConstrs((quicksum(sub_var_ac_rsc[(t,tp,m,d,k,c)] for tp in T) <= sub_var_st_ps[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C), name='con_resch')
sub_con_sched = sub_prob.addConstrs((quicksum(sub_var_ac_sc[(t,m,d,k,c)] for t in T) <= sub_var_st_pw[(m,d,k,c)] for m in M for d in D for k in K for c in C), name='con_sched')

sub_con_ul_bd = sub_prob.addConstrs((sub_var_st_ul[p] <= max(1, 2 * p_dat[p].expected_units) for p in P), name='con_ul_bound')
sub_con_pw_bd = sub_prob.addConstrs((sub_var_st_pw[(m,d,k,c)] <= max(1, 5*pea[(d,k,c)]) for m in M for d in D for k in K for c in C), name='con_pw_bound')
sub_con_ps_bd = sub_prob.addConstrs((sub_var_st_ps[(t,m,d,k,c)] <= max(1, 5*pea[(d,k,c)]) for t in T for m in M for d in D for k in K for c in C), name='con_pw_bound')

# sub_con_ul_bd = sub_prob.addConstrs((sub_var_st_ul[p] <= 20 for p in P), name='con_ul_bound')
# sub_con_pw_bd = sub_prob.addConstrs((sub_var_st_pw[(m,d,k,c)] <= 50 for m in M for d in D for k in K for c in C), name='con_pw_bound')
# sub_con_ps_bd = sub_prob.addConstrs((sub_var_st_ps[(t,m,d,k,c)] <= 50 for t in T for m in M for d in D for k in K for c in C), name='con_pw_bound')

#endregion

#%%
##### Phase 1 #####
#region

ma_obj_p2 = LinExpr()

# Solve Phase 1
iter = 0
while True:

    # Solve Master
    master.write(f'p1m{iter}.lp')
    master.optimize()
    print(f"PHASE 1 Master Iter {iter}:\t\t{master.ObjVal}")


    # Update Subproblem
    sub_obj_bul_co = quicksum( (sub_var_st_ul[p] - gam*sub_var_aux_ulp[p]) * ma_con_bul_co[p].Pi for p in PCO)
    sub_obj_bul_nco = quicksum(sub_var_st_ul[p] * ma_con_bul_nco[p].Pi for p in PNCO)

    sub_obj_bpw_0 = quicksum( (sub_var_st_pw[(0,d,k,c)] - gam*pea[(d,k,c)]) * ma_con_bpw_0[(0,d,k,c)].Pi for d in D for k in K for c in C)
    sub_obj_bpw_1TL = quicksum( (sub_var_st_pw[i] - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])]) * ma_con_bpw_1TL[i].Pi for i in mTLdkc)
    sub_obj_bpw_TLM = (
        quicksum( sub_var_st_pw[i] * ma_con_bpw_TLM[i].Pi for i in TLMdkc ) - 
        gam * quicksum( sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc ) -
        gam * quicksum( sub_var_aux_pwt_d[(i[0]-1, D[D.index(i[1])-1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[1] != D[0]) + 
        gam * quicksum( sub_var_aux_pwt_d[(i[0]-1, i[1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[1] != D[-1]) -
        gam * quicksum( sub_var_aux_pwt_k[(i[0]-1, i[1], K[K.index(i[2])-1],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[2] != K[0]) + 
        gam * quicksum( sub_var_aux_pwt_k[(i[0]-1, i[1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[2] != K[-1])    )
    sub_obj_bpw_M = (
        quicksum( sub_var_st_pw[(m,d,k,c)] * ma_con_bpw_M[(m,d,k,c)].Pi for m in M[-1:] for d in D for k in K for c in C ) - 
        gam * quicksum( sub_var_aux_pwp[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C ) -
        gam * quicksum( sub_var_aux_pwt_d[(m, d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if d != D[0]) + 
        gam * quicksum( sub_var_aux_pwt_d[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if d != D[-1]) -
        gam * quicksum( sub_var_aux_pwt_k[(m,d,K[K.index(k)-1],c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if k != K[0]) + 
        gam * quicksum( sub_var_aux_pwt_k[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if k != K[-1])    )

    sub_obj_bps_0 = quicksum( sub_var_st_ps[(t,0,d,k,c)] * ma_con_bps_0[(t,0,d,k,c)].Pi for t in T[:-1] for d in D for k in K for c in C )
    sub_obj_bps_1TL = quicksum( (sub_var_st_ps[i] - gam * sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])]) * ma_con_bps_1TL[i].Pi for i in tmTLdkc)
    sub_obj_bps_TLM = (
        quicksum( sub_var_st_ps[i] * ma_con_bps_TLM[i].Pi for i in tTLMdkc ) - 
        gam * quicksum( sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc ) -
        gam * quicksum( sub_var_aux_pst_d[(i[0]+1,i[1]-1,D[D.index(i[2])-1],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[2] != D[0]) + 
        gam * quicksum( sub_var_aux_pst_d[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[2] != D[-1]) -
        gam * quicksum( sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],K[K.index(i[3])-1],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[3] != K[0]) + 
        gam * quicksum( sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[3] != K[-1])    )
    sub_obj_bps_M = (
        quicksum( sub_var_st_ps[(t,m,d,k,c)] * ma_con_bps_M[(t,m,d,k,c)].Pi for t in T[:-1] for m in M[-1:] for d in D for k in K for c in C ) - 
        gam * quicksum( sub_var_aux_psp[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C ) -
        gam * quicksum( sub_var_aux_pst_d[(t+1,m,D[D.index(d)-1],k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if d != D[0]) + 
        gam * quicksum( sub_var_aux_pst_d[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if d != D[-1]) -
        gam * quicksum( sub_var_aux_pst_k[(t+1,m,d,K[K.index(k)-1],c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if k != K[0]) + 
        gam * quicksum( sub_var_aux_pst_k[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if k != K[-1])    )
    sub_obj_bps_T = quicksum( sub_var_st_ps[(T[-1],m,d,k,c)] * ma_con_bps_T[(T[-1],m,d,k,c)].Pi for m in M for d in D for k in K for c in C )
    sub_obj_val = (
        ((1-gam)*ma_con_b0.Pi) + 
        (sub_obj_bul_co + sub_obj_bul_nco) + 
        (sub_obj_bpw_0 + sub_obj_bpw_1TL + sub_obj_bpw_TLM + sub_obj_bpw_M) +
        (sub_obj_bps_0 + sub_obj_bps_1TL + sub_obj_bps_TLM + sub_obj_bps_M + sub_obj_bps_T))

    sub_obj_cw = quicksum( cw[k] * sub_var_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    sub_obj_cs = quicksum( cs[k][t] * sub_var_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    sub_obj_brsc = quicksum( (cs[k][tp-t]+cc[k]) * sub_var_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    sub_obj_grsc = quicksum( (cs[k][t-tp]-cc[k]) * sub_var_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    sub_obj_cv = quicksum( cv * sub_var_aux_uv[(t,p)] for  t in T for p in P)
    sub_obj_cost = (sub_obj_cw + sub_obj_cs + sub_obj_brsc - sub_obj_grsc + sub_obj_cv)

    sub_prob.setObjective( -sub_obj_val, GRB.MINIMIZE )


    # Solve Subproblem
    sub_prob.write(f'p1s{iter}.lp')
    sub_prob.optimize()


    # Update Master
    sa = Column()
    sa.addTerms(1-gam, ma_con_b0)
     
    [sa.addTerms(sub_var_st_ul[p].X - gam*sub_var_aux_ulp[p].X, ma_con_bul_co[p]) for p in PCO]    
    [sa.addTerms(sub_var_st_ul[p].X, ma_con_bul_nco[p]) for p in PNCO]

    [sa.addTerms(sub_var_st_pw[(0,d,k,c)].X - gam*pea[((d,k,c))], ma_con_bpw_0[0,d,k,c]) for d in D for k in K for c in C]
    [sa.addTerms(sub_var_st_pw[i].X - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])].X, ma_con_bpw_1TL[i]) for i in mTLdkc]
    [sa.addTerms(sub_var_st_pw[i].X - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc]
    [sa.addTerms(-gam*sub_var_aux_pwt_d[(i[0]-1, D[D.index(i[1])-1], i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[1] != D[0]]
    [sa.addTerms(gam*sub_var_aux_pwt_d[(i[0]-1, i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[1] != D[-1]]
    [sa.addTerms(-gam*sub_var_aux_pwt_k[(i[0]-1,i[1], K[K.index(i[2]-1)],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[2] != K[0]]
    [sa.addTerms(gam*sub_var_aux_pwt_k[(i[0]-1, i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[2] != K[-1]]
    [sa.addTerms(sub_var_st_pw[(M[-1],d,k,c)].X -gam*quicksum(sub_var_aux_pwp[(m,d,k,c)] for m in M[-2:]).getValue(), ma_con_bpw_M[M[-1],d,k,c]) for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum( sub_var_aux_pwt_d[(m,D[D.index(d)-1],k,c)] for m in M[-2:] if d != D[0]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum( sub_var_aux_pwt_d[(m,d,k,c)] for m in M[-2:] if d != D[-1]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum( sub_var_aux_pwt_k[(m,d,K[K.index(k)-1],c)] for m in M[-2:] if k != K[0]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum( sub_var_aux_pwt_k[(m,d,k,c)] for m in M[-2:] if k != K[-1]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]

    [sa.addTerms( sub_var_st_ps[(t,0,d,k,c)].X, ma_con_bps_0[(t,0,d,k,c)] ) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( sub_var_st_ps[i].X - gam*sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_1TL[i]) for i in tmTLdkc]
    [sa.addTerms( sub_var_st_ps[i].X - gam*sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc ]
    [sa.addTerms(-gam*sub_var_aux_pst_d[(i[0]+1,i[1]-1, D[D.index(i[2])-1], i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[2] != D[0]]
    [sa.addTerms(gam*sub_var_aux_pst_d[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[2] != D[-1]]
    [sa.addTerms(-gam*sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2], K[K.index(i[3]-1)],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[3] != K[0]]
    [sa.addTerms(gam*sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[3] != K[-1]]
    [sa.addTerms( sub_var_st_ps[(t,M[-1],d,k,c)].X -gam*quicksum(sub_var_aux_psp[(t+1,m,d,k,c)] for m in M[:-2]).getValue(), ma_con_bps_M[(t, M[-1], d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum(  sub_var_aux_pst_d[(t+1,m,D[D.index(d)-1],k,c)] for m in M[:-2] if d != D[0]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum(  sub_var_aux_pst_d[(t+1,m,d,k,c)] for m in M[:-2] if d != D[-1]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum(  sub_var_aux_pst_k[(t+1,m,d,K[K.index(k)-1],c)] for m in M[:-2] if k != K[0]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum(  sub_var_aux_pst_k[(t+1,m,d,k,c)] for m in M[:-2] if k != K[-1]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    
    [sa.addTerms( sub_var_st_ps[(T[-1],m,d,k,c)].X, ma_con_bps_T[(T[-1],m,d,k,c)] ) for m in M for d in D for k in K for c in C]

    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa)


    # Save objective for phase 2
    ma_obj_p2.add(sa_var, sub_obj_cost.getValue())


    # End Condition
    if master.ObjVal <= 0: 
        master.optimize()
        break

    iter += 1
#endregion    

# %%
##### Phase 2 #####
#region

# Update Master Model
master.remove(ma_var_b0)
master.remove(ma_var_bul_co)
master.remove(ma_var_bul_nco)
master.remove(ma_var_bpw_0)
master.remove(ma_var_bpw_1TL)
master.remove(ma_var_bpw_TLM)
master.remove(ma_var_bpw_M)
master.remove(ma_var_bps_0)
master.remove(ma_var_bps_1TL)
master.remove(ma_var_bps_TLM)
master.remove(ma_var_bps_M)
master.remove(ma_var_bps_T)
master.setObjective(ma_obj_p2, GRB.MINIMIZE)

# Solve Phase 2
iter += 1
while True:

    # Solve Master
    master.write(f'p2m{iter}.lp')
    master.optimize()
    # print(f"PHASE 2 Master Iter {iter}:\t\t{master.ObjVal}")
    

    # Update Subproblem
    sub_obj_bul_co = quicksum( (sub_var_st_ul[p] - gam*sub_var_aux_ulp[p]) * ma_con_bul_co[p].Pi for p in PCO)
    sub_obj_bul_nco = quicksum(sub_var_st_ul[p] * ma_con_bul_nco[p].Pi for p in PNCO)

    sub_obj_bpw_0 = quicksum( (sub_var_st_pw[(0,d,k,c)] - gam*pea[(d,k,c)]) * ma_con_bpw_0[(0,d,k,c)].Pi for d in D for k in K for c in C)
    sub_obj_bpw_1TL = quicksum( (sub_var_st_pw[i] - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])]) * ma_con_bpw_1TL[i].Pi for i in mTLdkc)
    sub_obj_bpw_TLM = (
        quicksum( sub_var_st_pw[i] * ma_con_bpw_TLM[i].Pi for i in TLMdkc ) - 
        gam * quicksum( sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc ) -
        gam * quicksum( sub_var_aux_pwt_d[(i[0]-1, D[D.index(i[1])-1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[1] != D[0]) + 
        gam * quicksum( sub_var_aux_pwt_d[(i[0]-1, i[1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[1] != D[-1]) -
        gam * quicksum( sub_var_aux_pwt_k[(i[0]-1, i[1], K[K.index(i[2])-1],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[2] != K[0]) + 
        gam * quicksum( sub_var_aux_pwt_k[(i[0]-1, i[1], i[2],i[3])] * ma_con_bpw_TLM[i].Pi for i in TLMdkc if i[2] != K[-1])    )
    sub_obj_bpw_M = (
        quicksum( sub_var_st_pw[(m,d,k,c)] * ma_con_bpw_M[(m,d,k,c)].Pi for m in M[-1:] for d in D for k in K for c in C ) - 
        gam * quicksum( sub_var_aux_pwp[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C ) -
        gam * quicksum( sub_var_aux_pwt_d[(m, d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if d != D[0]) + 
        gam * quicksum( sub_var_aux_pwt_d[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if d != D[-1]) -
        gam * quicksum( sub_var_aux_pwt_k[(m,d,K[K.index(k)-1],c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if k != K[0]) + 
        gam * quicksum( sub_var_aux_pwt_k[(m,d,k,c)] * ma_con_bpw_M[(M[-1],d,k,c)].Pi for m in M[-2:] for d in D for k in K for c in C if k != K[-1])    )

    sub_obj_bps_0 = quicksum( sub_var_st_ps[(t,0,d,k,c)] * ma_con_bps_0[(t,0,d,k,c)].Pi for t in T[:-1] for d in D for k in K for c in C )
    sub_obj_bps_1TL = quicksum( (sub_var_st_ps[i] - gam * sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])]) * ma_con_bps_1TL[i].Pi for i in tmTLdkc)
    sub_obj_bps_TLM = (
        quicksum( sub_var_st_ps[i] * ma_con_bps_TLM[i].Pi for i in tTLMdkc ) - 
        gam * quicksum( sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc ) -
        gam * quicksum( sub_var_aux_pst_d[(i[0]+1,i[1]-1,D[D.index(i[2])-1],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[2] != D[0]) + 
        gam * quicksum( sub_var_aux_pst_d[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[2] != D[-1]) -
        gam * quicksum( sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],K[K.index(i[3])-1],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[3] != K[0]) + 
        gam * quicksum( sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],i[3],i[4])] * ma_con_bps_TLM[i].Pi for i in tTLMdkc if i[3] != K[-1])    )
    sub_obj_bps_M = (
        quicksum( sub_var_st_ps[(t,m,d,k,c)] * ma_con_bps_M[(t,m,d,k,c)].Pi for t in T[:-1] for m in M[-1:] for d in D for k in K for c in C ) - 
        gam * quicksum( sub_var_aux_psp[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C ) -
        gam * quicksum( sub_var_aux_pst_d[(t+1,m,D[D.index(d)-1],k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if d != D[0]) + 
        gam * quicksum( sub_var_aux_pst_d[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if d != D[-1]) -
        gam * quicksum( sub_var_aux_pst_k[(t+1,m,d,K[K.index(k)-1],c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if k != K[0]) + 
        gam * quicksum( sub_var_aux_pst_k[(t+1,m,d,k,c)] * ma_con_bps_M[(t,M[-1],d,k,c)].Pi for t in T[:-1] for m in M[-2:] for d in D for k in K for c in C if k != K[-1])    )
    sub_obj_bps_T = quicksum( sub_var_st_ps[(T[-1],m,d,k,c)] * ma_con_bps_T[(T[-1],m,d,k,c)].Pi for m in M for d in D for k in K for c in C )
    sub_obj_val = (
        ((1-gam)*ma_con_b0.Pi) + 
        (sub_obj_bul_co + sub_obj_bul_nco) + 
        (sub_obj_bpw_0 + sub_obj_bpw_1TL + sub_obj_bpw_TLM + sub_obj_bpw_M) +
        (sub_obj_bps_0 + sub_obj_bps_1TL + sub_obj_bps_TLM + sub_obj_bps_M + sub_obj_bps_T))

    sub_obj_cw = quicksum( cw[k] * sub_var_aux_pwp[(m,d,k,c)] for m in M for d in D for k in K for c in C )
    sub_obj_cs = quicksum( cs[k][t] * sub_var_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    sub_obj_brsc = quicksum( (cs[k][tp-t]+cc[k]) * sub_var_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp > t)
    sub_obj_grsc = quicksum( (cs[k][t-tp]-cc[k]) * sub_var_ac_rsc[(t,tp,m,d,k,c)] for t in T for tp in T for m in M for d in D for k in K for c in C if tp < t)
    sub_obj_cv = quicksum( cv * sub_var_aux_uv[(t,p)] for  t in T for p in P)
    sub_obj_cost = (sub_obj_cw + sub_obj_cs + sub_obj_brsc - sub_obj_grsc + sub_obj_cv)

    sub_prob.setObjective( sub_obj_cost-sub_obj_val, GRB.MINIMIZE )


    # Solve Subproblem
    sub_prob.write(f'p2s{iter}.lp')
    sub_prob.optimize()
    print(f"PHASE 2 Sub Iter {iter}:\t\t{sub_prob.ObjVal}")


    # Update Master
    sa = Column()
    sa.addTerms(1-gam, ma_con_b0)
     
    [sa.addTerms(sub_var_st_ul[p].X - gam*sub_var_aux_ulp[p].X, ma_con_bul_co[p]) for p in PCO]    
    [sa.addTerms(sub_var_st_ul[p].X, ma_con_bul_nco[p]) for p in PNCO]

    [sa.addTerms(sub_var_st_pw[(0,d,k,c)].X - gam*pea[((d,k,c))], ma_con_bpw_0[0,d,k,c]) for d in D for k in K for c in C]
    [sa.addTerms(sub_var_st_pw[i].X - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])].X, ma_con_bpw_1TL[i]) for i in mTLdkc]
    [sa.addTerms(sub_var_st_pw[i].X - gam*sub_var_aux_pwp[(i[0]-1,i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc]
    [sa.addTerms(-gam*sub_var_aux_pwt_d[(i[0]-1, D[D.index(i[1])-1], i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[1] != D[0]]
    [sa.addTerms(gam*sub_var_aux_pwt_d[(i[0]-1, i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[1] != D[-1]]
    [sa.addTerms(-gam*sub_var_aux_pwt_k[(i[0]-1,i[1], K[K.index(i[2]-1)],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[2] != K[0]]
    [sa.addTerms(gam*sub_var_aux_pwt_k[(i[0]-1, i[1],i[2],i[3])].X, ma_con_bpw_TLM[i]) for i in TLMdkc if i[2] != K[-1]]
    [sa.addTerms(sub_var_st_pw[(M[-1],d,k,c)].X -gam*quicksum(sub_var_aux_pwp[(m,d,k,c)] for m in M[-2:]).getValue(), ma_con_bpw_M[M[-1],d,k,c]) for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum( sub_var_aux_pwt_d[(m,D[D.index(d)-1],k,c)] for m in M[-2:] if d != D[0]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum( sub_var_aux_pwt_d[(m,d,k,c)] for m in M[-2:] if d != D[-1]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum( sub_var_aux_pwt_k[(m,d,K[K.index(k)-1],c)] for m in M[-2:] if k != K[0]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum( sub_var_aux_pwt_k[(m,d,k,c)] for m in M[-2:] if k != K[-1]).getValue(), ma_con_bpw_M[(M[-1],d,k,c)] ) for d in D for k in K for c in C]

    [sa.addTerms( sub_var_st_ps[(t,0,d,k,c)].X, ma_con_bps_0[(t,0,d,k,c)] ) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( sub_var_st_ps[i].X - gam*sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_1TL[i]) for i in tmTLdkc]
    [sa.addTerms( sub_var_st_ps[i].X - gam*sub_var_aux_psp[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc ]
    [sa.addTerms(-gam*sub_var_aux_pst_d[(i[0]+1,i[1]-1, D[D.index(i[2])-1], i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[2] != D[0]]
    [sa.addTerms(gam*sub_var_aux_pst_d[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[2] != D[-1]]
    [sa.addTerms(-gam*sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2], K[K.index(i[3]-1)],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[3] != K[0]]
    [sa.addTerms(gam*sub_var_aux_pst_k[(i[0]+1,i[1]-1,i[2],i[3],i[4])].X, ma_con_bps_TLM[i]) for i in tTLMdkc if i[3] != K[-1]]
    [sa.addTerms( sub_var_st_ps[(t,M[-1],d,k,c)].X -gam*quicksum(sub_var_aux_psp[(t+1,m,d,k,c)] for m in M[:-2]).getValue(), ma_con_bps_M[(t, M[-1], d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum(  sub_var_aux_pst_d[(t+1,m,D[D.index(d)-1],k,c)] for m in M[:-2] if d != D[0]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum(  sub_var_aux_pst_d[(t+1,m,d,k,c)] for m in M[:-2] if d != D[-1]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( -gam*quicksum(  sub_var_aux_pst_k[(t+1,m,d,K[K.index(k)-1],c)] for m in M[:-2] if k != K[0]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( gam*quicksum(  sub_var_aux_pst_k[(t+1,m,d,k,c)] for m in M[:-2] if k != K[-1]).getValue(), ma_con_bps_M[(t,M[-1],d,k,c)]) for t in T[:-1] for d in D for k in K for c in C]
    [sa.addTerms( sub_var_st_ps[(T[-1],m,d,k,c)].X, ma_con_bps_T[(T[-1],m,d,k,c)] ) for m in M for d in D for k in K for c in C]

    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa, obj=sub_obj_cost.getValue())


    # End Condition
    if sub_prob.ObjVal >= 0:
        master.optimize()
        break
    
    if iter >= 100:
        break

    # Trims
    # if iter %10 == 0:
    #     for i in master.getVars():
    #         if i.X == 0:
    #             master.remove(i)


    iter += 1
#endregion



    
print(f'\tBeta 0: {ma_con_b0.Pi}')

for p in PCO: print(f'\tBeta UL - {p}: {ma_con_bul_co[p].Pi}')
for p in PNCO: print(f'\tBeta UL - {p}: {ma_con_bul_nco[p].Pi}')

for i in itertools.product(M[0:1], D, K, C): print(f'\tBeta PW - {i}: {ma_con_bpw_0[i].Pi}')
for i in mTLdkc: print(f'\tBeta PW - {i}: {ma_con_bpw_1TL[i].Pi}')
for i in TLMdkc: print(f'\tBeta PW - {i}: {ma_con_bpw_TLM[i].Pi}')
for i in itertools.product(M[-1:], D, K, C): print(f'\tBeta PW - {i}: {ma_con_bpw_M[i].Pi}')

for i in itertools.product(T[:-1], M[0:1], D, K, C): print(f'\tBeta PS - {i}: {ma_con_bps_0[i].Pi}')
for i in tmTLdkc: print(f'\tBeta PS - {i}: {ma_con_bps_1TL[i].Pi}')
for i in tTLMdkc: print(f'\tBeta PS - {i}: {ma_con_bps_TLM[i].Pi}')
for i in itertools.product(T[:-1], M[-1:], D, K, C): print(f'\tBeta PS - {i}: {ma_con_bps_M[i].Pi}')
for i in itertools.product(T[-1:], M, D, K, C): print(f'\tBeta PS - {i}: {ma_con_bps_T[i].Pi}')

# %%
