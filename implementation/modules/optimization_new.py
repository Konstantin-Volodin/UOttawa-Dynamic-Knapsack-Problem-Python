##### Initialization & Changeable Parameters ######
#region
from modules import data_import
import numpy as np
import gurobipy as gp
import itertools
import os.path
import pickle
#endregion


class optimization_handler:
    
    ##### Initialize #####
    def __init__(self, optim_params, optim_paths, import_model=False):
        self.iter_lims = optim_params['iterations']
        self.beta_fun = optim_params['beta_function']
        self.sub_mip_gap = optim_params['subproblem_mip_gap']
        self.import_data = optim_paths['import_params']
        self.export_data = optim_paths['export_betas']
        self.export_model = optim_paths['export_model']
        self.import_model = import_model
        self.my_path = os.getcwd()
        

    ##### Read Data #####
    def read_data(self):
        self.input_data = data_import.read_data(os.path.join(self.my_path, self.import_data))

        # Quick Access to Various Parameters
        self.TL = self.input_data.transition.wait_limit
        self.BM = 10000
        self.U = self.input_data.usage
        self.p_dat = self.input_data.ppe_data
        self.pea = self.input_data.arrival
        self.gam = self.input_data.model_param.gamma
        self.ptp_d = self.input_data.transition.transition_rate_comp
        self.ptp_k = self.input_data.transition.transition_rate_pri
        self.cw = self.input_data.model_param.cw
        self.cs = self.input_data.model_param.cs
        self.cc = self.input_data.model_param.cc
        self.cv = self.input_data.model_param.cv
        self.cuu = 1000

        ##### Generating Sets #####
        self.T = self.input_data.indices['t']
        self.M = self.input_data.indices['m']
        self.P = self.input_data.indices['p']
        self.D = self.input_data.indices['d']
        self.K = self.input_data.indices['k']
        self.C = self.input_data.indices['c']

        # Sub Sets
        self.PCO = []
        self.PNCO = []
        for k, v in self.input_data.ppe_data.items(): 
            if v.ppe_type == "non-carry-over": self.PNCO.append(k) 
            else: self.PCO.append(k)
            
        self.mTLdkc = [(m, d, k, c) for c in self.C for m in self.M[1:self.input_data.transition.wait_limit[c]] for d in self.D for k in self.K ]
        self.TLMdkc = [(m, d, k, c) for c in self.C for m in self.M[self.input_data.transition.wait_limit[c]:-1] for d in self.D for k in self.K ]
        self.tmTLdkc = [(t, m, d, k, c) for t in self.T[:-1] for c in self.C for m in self.M[1:self.input_data.transition.wait_limit[c]] for d in self.D for k in self.K ]
        self.tTLMdkc = [(t, m, d, k, c) for t in self.T[:-1] for c in self.C for m in self.M[self.input_data.transition.wait_limit[c]:-1] for d in self.D for k in self.K ]

        # Expected Data 
        self.E_UL = self.input_data.expected_state_values['ul']
        self.E_PW = self.input_data.expected_state_values['pw']
        self.E_PS = self.input_data.expected_state_values['ps']


    ##### Master Model #####
    def generate_master(self):
        self.master = gp.Model("Master problem")
        self.master.params.LogToConsole = 0

        # Goal Variables
        self.mv_b0 = self.master.addVar(vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_0')

        self.mv_bul_co = self.master.addVars(self.PCO, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ul_co')
        self.mv_bul_nco = self.master.addVars(self.PNCO, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ul_nco')

        self.mv_bpw_0 = self.master.addVars(self.M[0:1], self.D, self.K, self.C, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_pw_0dkc')
        self.mv_bpw_1TL = self.master.addVars(self.mTLdkc, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_pw_1TLdkc')
        self.mv_bpw_TLM = self.master.addVars(self.TLMdkc, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_pw_TLMdkc')
        self.mv_bpw_M = self.master.addVars(self.M[-1:], self.D, self.K, self.C, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_pw_Mdkc')

        self.mv_bps_0 = self.master.addVars(self.T[:-1], self.M[0:1], self.D, self.K, self.C, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ps_t0dkc')
        self.mv_bps_1TL = self.master.addVars(self.tmTLdkc, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ps_t1TLdkc')
        self.mv_bps_TLM = self.master.addVars(self.tTLMdkc, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ps_tTLMdkc')
        self.mv_bps_M = self.master.addVars(self.T[:-1], self.M[-1:], self.D, self.K, self.C, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ps_tMdkc')
        self.mv_bps_T = self.master.addVars(self.T[-1:], self.M, self.D, self.K, self.C, vtype = gp.GRB.CONTINUOUS, lb=0, name='var_beta_ps_Tmdkc')

        # Constraint Definition
        self.mc_b0 = self.master.addConstr(self.mv_b0 == 1, name='con_beta_0')

        self.mc_bul_co = self.master.addConstrs((self.mv_bul_co[p] >= self.E_UL[p] for p in self.PCO), name='con_beta_ul_co')
        self.mc_bul_nco = self.master.addConstrs((self.mv_bul_nco[p] >= self.E_UL[p] for p in self.PNCO), name='con_beta_ul_nco')

        self.mc_bpw_0 = self.master.addConstrs((self.mv_bpw_0[(m, d, k, c)] >= self.E_PW[(m, d, k, c)] for m in self.M[0:1] for d in self.D for k in self.K for c in self.C), name='con_beta_pw_0dkc')
        self.mc_bpw_1TL = self.master.addConstrs((self.mv_bpw_1TL[i] >= self.E_PW[i] for i in self.mTLdkc), name='con_beta_pw_1TLdkc')
        self.mc_bpw_TLM = self.master.addConstrs((self.mv_bpw_TLM[i] >= self.E_PW[i] for i in self.TLMdkc), name='con_beta_pw_TLMdkc')
        self.mc_bpw_M = self.master.addConstrs((self.mv_bpw_M[(m, d, k, c)] >= self.E_PW[(m, d, k, c)] for m in self.M[-1:] for d in self.D for k in self.K for c in self.C), name='con_beta_pw_Mdkc')

        self.mc_bps_0 = self.master.addConstrs((self.mv_bps_0[(t, m, d, k, c)] >= self.E_PS[(t, m, d, k, c)] for t in self.T[:-1] for m in self.M[0:1] for d in self.D for k in self.K for c in self.C), name='con_beta_ps_t0dkc')
        self.mc_bps_1TL = self.master.addConstrs((self.mv_bps_1TL[i] >= self.E_PS[i] for i in self.tmTLdkc), name='con_beta_ps_t1TLdkc')
        self.mc_bps_TLM = self.master.addConstrs((self.mv_bps_TLM[i] >= self.E_PS[i] for i in self.tTLMdkc), name='con_beta_ps_tTLMdkc')
        self.mc_bps_M = self.master.addConstrs((self.mv_bps_M[(t, m, d, k, c)] >= self.E_PS[(t, m, d, k, c)] for t in self.T[:-1] for m in self.M[-1:] for d in self.D for k in self.K for c in self.C ), name='con_beta_ps_tMdkc')
        self.mc_bps_T = self.master.addConstrs((self.mv_bps_T[(t, m, d, k, c)] >= self.E_PS[(t, m, d, k, c)] for t in self.T[-1:] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_beta_ps_Tmdkc')

        # Objective Function Definition
        self.mo_bul_co = gp.quicksum(self.mv_bul_co[p] for p in self.PCO)
        self.mo_bul_nco = gp.quicksum(self.mv_bul_nco[p] for p in self.PNCO)

        self.mo_bpw_0 = gp.quicksum(self.mv_bpw_0[(0, d, k, c)] for d in self.D for k in self.K for c in self.C)
        self.mo_bpw_1TL = gp.quicksum(self.mv_bpw_1TL[i] for i in self.mTLdkc)
        self.mo_bpw_TLM = gp.quicksum(self.mv_bpw_TLM[i] for i in self.TLMdkc)
        self.mo_bpw_M = gp.quicksum(self.mv_bpw_M[(self.M[-1], d, k, c)] for d in self.D for k in self.K for c in self.C)

        self.mo_bps_0 = gp.quicksum(self.mv_bps_0[(t, 0, d, k, c)] for t in self.T[:-1] for d in self.D for k in self.K for c in self.C)
        self.mo_bps_1TL = gp.quicksum(self.mv_bps_1TL[i] for i in self.tmTLdkc)
        self.mo_bps_TLM = gp.quicksum(self.mv_bps_TLM[i] for i in self.tTLMdkc)
        self.mo_bps_M = gp.quicksum(self.mv_bps_M[t, self.M[-1], d, k, c] for t in self.T[:-1] for d in self.D for k in self.K for c in self.C)
        self.mo_bps_T = gp.quicksum(self.mv_bps_T[self.T[-1], m, d, k, c] for m in self.M for d in self.D for k in self.K for c in self.C)

        self.master.setObjective( self.mv_b0 + (self.mo_bul_co + self.mo_bul_nco) + (self.mo_bpw_0 + self.mo_bpw_1TL + self.mo_bpw_TLM + self.mo_bpw_M) + (self.mo_bps_0 + self.mo_bps_1TL + self.mo_bps_TLM + self.mo_bps_M + self.mo_bps_T), gp.GRB.MINIMIZE)

    
    ##### Sub Problem #####
    def generate_subproblem(self):
        self.sub_prob = gp.Model('Sub problem')
        self.sub_prob.params.LogToConsole = 0
        self.sub_prob.params.MIPGap = self.sub_mip_gap

        # State Action & Auxiliary Variables
        self.sv_st_ul = self.sub_prob.addVars(self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_state_ul')
        self.sv_st_pw = self.sub_prob.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_pw')
        self.sv_st_ps = self.sub_prob.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_ps')

        self.sv_ac_sc = self.sub_prob.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_sc')
        self.sv_ac_rsc = self.sub_prob.addVars(self.T, self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_rsc')

        self.sv_aux_uv = self.sub_prob.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
        self.sv_aux_uvb = self.sub_prob.addVars(self.T, self.P, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

        self.sv_aux_ulp = self.sub_prob.addVars(self.PCO, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
        self.sv_aux_ulb = self.sub_prob.addVars(self.PCO, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

        self.sv_aux_uup = self.sub_prob.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
        self.sv_aux_pwp = self.sub_prob.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
        self.sv_aux_psp = self.sub_prob.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

        self.sv_aux_pwt_d = self.sub_prob.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
        self.sv_aux_pwt_k = self.sub_prob.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
        self.sv_aux_pst_d = self.sub_prob.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
        self.sv_aux_pst_k = self.sub_prob.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

        self.sv_aux_uu = self.sub_prob.addVars(self.T, vtype=gp.GRB.CONTINUOUS, lb=0, name='var_auxiliary_uu')
        self.sv_aux_uub = self.sub_prob.addVars(self.T, vtype=gp.GRB.BINARY, lb=0, name='var_auxiliary_uub')

        # Definition of auxiliary variables
        self.sc_uup = self.sub_prob.addConstrs((self.sv_aux_uup[(t,p)] == gp.quicksum( self.U[(p,d,c)] * self.sv_aux_psp[(t,m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C) for t in self.T for p in self.P), name='con_auxiliary_uup')
        self.sc_pwp = self.sub_prob.addConstrs((self.sv_aux_pwp[(m,d,k,c)] == self.sv_st_pw[(m,d,k,c)] - gp.quicksum( self.sv_ac_sc[(t,m,d,k,c)] for t in self.T) for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwp')
        self.sc_psp = self.sub_prob.addConstrs((self.sv_aux_psp[(t,m,d,k,c)] == self.sv_st_ps[(t,m,d,k,c)] + self.sv_ac_sc[(t,m,d,k,c)] + gp.quicksum( self.sv_ac_rsc[tp,t,m,d,k,c] for tp in self.T) - gp.quicksum( self.sv_ac_rsc[t,tp,m,d,k,c] for tp in self.T) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pws')

        self.sc_aux_uv_0 = self.sub_prob.addConstrs((self.sv_aux_uv[(t,p)] >= 0 for t in self.T for p in self.P), name='con_auxiliary_uv_0')
        self.sc_aux_uv_0M = self.sub_prob.addConstrs((self.sv_aux_uv[(t,p)] <= self.BM * self.sv_aux_uvb[(t,p)] for t in self.T for p in self.P), name='con_auxiliary_uv_0M')
        self.sc_aux_uv_1 = self.sub_prob.addConstrs((self.sv_aux_uv[(1,p)] >= self.sv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.sv_st_ul[p] for p in self.P), name='con_auxiliary_uv_1')
        self.sc_aux_uv_1M = self.sub_prob.addConstrs((self.sv_aux_uv[(1,p)] <= (self.sv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.sv_st_ul[p]) + self.BM * (1 - self.sv_aux_uvb[(1, p)]) for p in self.P), name='con_auxiliary_uv_1M')
        self.sc_aux_uv_m = self.sub_prob.addConstrs((self.sv_aux_uv[(t, p)] >= (self.sv_aux_uup[(t, p)] - self.p_dat[p].expected_units) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_m')
        self.sc_aux_uv_mM = self.sub_prob.addConstrs((self.sv_aux_uv[(t, p)] <= (self.sv_aux_uup[(t,p)] - self.p_dat[p].expected_units) + self.BM * (1 - self.sv_aux_uvb[(t, p)]) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_mM')

        self.sc_aux_ulp_0 = self.sub_prob.addConstrs((self.sv_aux_ulp[p] >= 0 for p in self.PCO), name='con_auxiliary_ulp_0')
        self.sc_aux_ulp_0M = self.sub_prob.addConstrs((self.sv_aux_ulp[p] <= self.BM * self.sv_aux_ulb[p] for p in self.PCO), name='con_auxiliary_ulp_0M')
        self.sc_aux_ulp_p = self.sub_prob.addConstrs((self.sv_aux_ulp[p] >= (self.p_dat[p].expected_units + self.sv_st_ul[p] - self.sv_aux_uup[(1,p)]) for p in self.PCO), name='con_auxiliary_ulp_p')
        self.sc_aux_ulp_pM = self.sub_prob.addConstrs((self.sv_aux_ulp[p] <= (self.p_dat[p].expected_units + self.sv_st_ul[p] - self.sv_aux_uup[(1,p)]) + self.BM * (1-self.sv_aux_ulb[p]) for p in self.PCO), name='con_auxiliary_ulp_pM')

        self.sc_aux_pwt_d = self.sub_prob.addConstrs((self.sv_aux_pwt_d[(m,d,k,c)] == self.ptp_d[(d,c)] * self.sv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwt_d')
        self.sc_aux_pwt_k_0 = self.sub_prob.addConstrs((self.sv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_pwp[(m,d,k,c)] - self.sv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pwt_k_0')
        self.sc_aux_pwt_k_i = self.sub_prob.addConstrs((self.sv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_pwp[(m,d,k,c)] + self.sv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)] - self.sv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pwt_k')
        self.sc_aux_pwt_k_D = self.sub_prob.addConstrs((self.sv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_pwp[(m,d,k,c)] + self.sv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pwt_k_D')
        self.sc_aux_pwt_k = {**self.sc_aux_pwt_k_0, **self.sc_aux_pwt_k_i, **self.sc_aux_pwt_k_D}

        self.sc_aux_pst_d = self.sub_prob.addConstrs((self.sv_aux_pst_d[(t,m,d,k,c)] == self.ptp_d[(d,c)] * self.sv_aux_psp[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pst_d')
        self.sc_aux_pst_k_0 = self.sub_prob.addConstrs((self.sv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_psp[(t,m,d,k,c)] - self.sv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pwt_k_0')
        self.sc_aux_pst_k_i = self.sub_prob.addConstrs((self.sv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_psp[(t,m,d,k,c)] + self.sv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)] - self.sv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pwt_k')
        self.sc_aux_pst_k_D = self.sub_prob.addConstrs((self.sv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.sv_aux_psp[(t,m,d,k,c)] + self.sv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pwt_k_D')
        self.sc_aux_pst_k = {**self.sc_aux_pst_k_0, **self.sc_aux_pst_k_i, **self.sc_aux_pst_k_D}

        self.sc_aux_uu_0 = self.sub_prob.addConstrs((self.sv_aux_uu[(t)] >= 0 for t in self.T), name='con_auxiliary_uu_0')
        self.sc_aux_uu_0M = self.sub_prob.addConstrs((self.sv_aux_uu[(t)] <= self.BM * self.sv_aux_uub[(t)] for t in self.T), name='con_auxiliary_uu_0M')
        self.sc_aux_uu_1 = self.sub_prob.addConstr((self.sv_aux_uu[(1)] >= self.p_dat[self.P[0]].expected_units + self.sv_st_ul[self.P[0]] - self.sv_aux_uup[1,self.P[0]]), name='con_auxiliary_uu_1')
        self.sc_aux_uu_1M = self.sub_prob.addConstr((self.sv_aux_uu[(1)] <= (self.p_dat[self.P[0]].expected_units + self.sv_st_ul[self.P[0]] - self.sv_aux_uup[1,self.P[0]]) + self.BM*(1 - self.sv_aux_uub[(1)])), name='con_auxiliary_uu_1M')
        self.sc_aux_uu_m = self.sub_prob.addConstrs((self.sv_aux_uu[(t)] >= (self.p_dat[self.P[0]].expected_units + self.sv_aux_uup[t,self.P[0]]) for t in self.T[1:]), name='con_auxiliary_uu_m')
        self.sc_aux_uu_mM = self.sub_prob.addConstrs((self.sv_aux_uu[(t)] <= (self.p_dat[self.P[0]].expected_units + self.sv_aux_uup[t,self.P[0]]) + self.BM * (1 - self.sv_aux_uub[(t)]) for t in self.T[1:]), name='con_auxiliary_uu_mM')

        # State Action Constraints
        self.sc_usage_1 = self.sub_prob.addConstrs((self.sv_aux_uup[(1,p)] <= self.p_dat[p].expected_units + self.sv_st_ul[p] + self.sv_aux_uv[(1,p)] for p in self.P), name='con_usage_1')
        self.sc_usage_tT = self.sub_prob.addConstrs((self.sv_aux_uup[(t,p)] <= self.p_dat[p].expected_units + self.sv_aux_uv[(t,p)] for t in self.T[1:] for p in self.P), name='con_usage_tT')

        self.sc_rescb = self.sub_prob.addConstrs((self.sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T[1:] for tp in self.T[1:] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_reschedule_bounds')
        self.sc_rescb_ttp = self.sub_prob.addConstrs((self.sv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if t == tp == 1), name='con_reschedule_bounds_ttp')

        self.sc_resch = self.sub_prob.addConstrs((gp.quicksum(self.sv_ac_rsc[(t,tp,m,d,k,c)] for tp in self.T) <= self.sv_st_ps[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_resch')
        self.sc_sched = self.sub_prob.addConstrs((gp.quicksum(self.sv_ac_sc[(t,m,d,k,c)] for t in self.T) <= self.sv_st_pw[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_sched')

        self.sc_ul_bd = self.sub_prob.addConstrs((self.sv_st_ul[p] <= max(10, 2 * self.p_dat[p].expected_units) for p in self.P), name='con_ul_bound')
        self.sc_pw_bd = self.sub_prob.addConstrs((self.sv_st_pw[(m,d,k,c)] <= max(10, 5*self.pea[(d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C), name='con_pw_bound')
        self.sc_ps_bd = self.sub_prob.addConstrs((self.sv_st_ps[(t,m,d,k,c)] <= max(10, 5*self.pea[(d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_pw_bound')


    ##### Phase 1 #####
    def solve_phase1(self):
        self.mo_p2 = gp.LinExpr()

        # Solve Phase 1
        self.iter = 0
        while True and self.import_model == False:

            # Solve Master
            self.master.optimize()
            print(f"PHASE 1 Master Iter {self.iter}:\t\t{self.master.ObjVal}")


            # Generate Value Equations
            val_b0 = gp.LinExpr(1-self.gam)

            val_bul_co = {p: gp.LinExpr(self.sv_st_ul[p] - self.gam*self.sv_aux_ulp[p])  for p in self.PCO }
            val_bul_nco = {p: gp.LinExpr(self.sv_st_ul[p])  for p in self.PNCO }

            val_bpw_0 = {(0,d,k,c): gp.LinExpr(self.sv_st_pw[(0,d,k,c)] - self.gam*self.pea[(d,k,c)])  for d in self.D for k in self.K for c in self.C}
            val_bpw_1TL = {(m,d,k,c): gp.LinExpr(self.sv_st_pw[(m,d,k,c)] - self.gam*self.sv_aux_pwp[(m-1,d,k,c)])  for m,d,k,c in self.mTLdkc}
            val_bpw_TLM = {}
            for m,d,k,c in self.TLMdkc:
                val_bpw_TLM[(m,d,k,c)] = gp.LinExpr( self.sv_st_pw[(m,d,k,c)] - self.gam*self.sv_aux_pwp[(m-1,d,k,c)] )
                if d != self.D[0]:  val_bpw_TLM[(m,d,k,c)] += -self.gam * self.sv_aux_pwt_d[(m-1, self.D[self.D.index(d)-1] ,k,c)]
                if k != self.K[0]:  val_bpw_TLM[(m,d,k,c)] += -self.gam * self.sv_aux_pwt_k[(m-1,d, self.K[self.K.index(k)-1] ,c)]
                if d != self.D[-1]: val_bpw_TLM[(m,d,k,c)] +=  self.gam * self.sv_aux_pwt_d[(m-1,d,k,c)]
                if k != self.K[-1]: val_bpw_TLM[(m,d,k,c)] +=  self.gam * self.sv_aux_pwt_k[(m-1,d,k,c)]
            val_bpw_M = {}
            for m,d,k,c in itertools.product( self.M[-1:], self.D, self.K, self.C):
                val_bpw_M[(m,d,k,c)] = gp.LinExpr( self.sv_st_pw[(m,d,k,c)] - self.gam*gp.quicksum( self.sv_aux_pwp[mm,d,k,c] for mm in self.M[-2:] ) )
                if d != self.D[0]:  val_bpw_M[(m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pwt_d[(mm, self.D[self.D.index(d)-1] ,k,c)] for mm in self.M[-2:])
                if k != self.K[0]:  val_bpw_M[(m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pwt_k[(mm,d, self.K[self.K.index(k)-1] ,c)] for mm in self.M[-2:])
                if d != self.D[-1]: val_bpw_M[(m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pwt_d[(mm,d,k,c)] for mm in self.M[-2:])
                if k != self.K[-1]: val_bpw_M[(m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pwt_k[(mm,d,k,c)] for mm in self.M[-2:])    

            val_bps_0 = {(t,0,d,k,c): gp.LinExpr(self.sv_st_ps[(t,0,d,k,c)])  for t in self.T[:-1] for d in self.D for k in self.K for c in self.C }
            val_bps_1TL = {(t,m,d,k,c): gp.LinExpr(self.sv_st_ps[(t,m,d,k,c)] - self.gam*self.sv_aux_psp[(t+1,m-1,d,k,c)]) for t,m,d,k,c in self.tmTLdkc}
            val_bps_TLM = {}
            for t,m,d,k,c in self.tTLMdkc:
                val_bps_TLM[(t,m,d,k,c)] = gp.LinExpr( self.sv_st_ps[(t,m,d,k,c)] - self.gam*self.sv_aux_psp[(t+1,m-1,d,k,c)] )
                if d != self.D[0]:  val_bps_TLM[(t,m,d,k,c)] += -self.gam * self.sv_aux_pst_d[(t+1,m-1, self.D[self.D.index(d)-1] ,k,c)]
                if k != self.K[0]:  val_bps_TLM[(t,m,d,k,c)] += -self.gam * self.sv_aux_pst_k[(t+1,m-1,d, self.K[self.K.index(k)-1] ,c)]
                if d != self.D[-1]: val_bps_TLM[(t,m,d,k,c)] +=  self.gam * self.sv_aux_pst_d[(t+1,m-1,d,k,c)]
                if k != self.K[-1]: val_bps_TLM[(t,m,d,k,c)] +=  self.gam * self.sv_aux_pst_k[(t+1,m-1,d,k,c)]
            val_bps_M = {}
            for t,m,d,k,c in itertools.product(self.T[:-1], self.M[-1:], self.D, self.K, self.C):
                val_bps_M[(t,m,d,k,c)] = gp.LinExpr( self.sv_st_ps[(t,m,d,k,c)] - self.gam*gp.quicksum( self.sv_aux_psp[(t+1,mm,d,k,c)] for mm in self.M[-2:] ) )
                if d != self.D[0]:  val_bps_M[(t,m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pst_d[(t+1,mm, self.D[self.D.index(d)-1] ,k,c)] for mm in self.M[-2:])
                if k != self.K[0]:  val_bps_M[(t,m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pst_k[(t+1,mm,d, self.K[self.K.index(k)-1] ,c)] for mm in self.M[-2:])
                if d != self.D[-1]: val_bps_M[(t,m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pst_d[(t+1,mm,d,k,c)] for mm in self.M[-2:])
                if k != self.K[-1]: val_bps_M[(t,m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pst_k[(t+1,mm,d,k,c)] for mm in self.M[-2:])    
            val_bps_T = {(self.T[-1],m,d,k,c): gp.LinExpr( self.sv_st_ps[(self.T[-1],m,d,k,c)] )  for m in self.M for d in self.D for k in self.K for c in self.C}

            # Update Subproblem
            so_val =   ((   self.mc_b0.Pi * val_b0                                                       ) +
                        (   gp.quicksum(self.mc_bul_co[i].Pi*val_bul_co[i]      for i in self.mc_bul_co)     + 
                            gp.quicksum(self.mc_bul_nco[i].Pi*val_bul_nco[i]    for i in self.mc_bul_nco)        ) + 
                        (   gp.quicksum(self.mc_bpw_0[i].Pi*val_bpw_0[i]        for i in self.mc_bpw_0)      + 
                            gp.quicksum(self.mc_bpw_1TL[i].Pi*val_bpw_1TL[i]    for i in self.mc_bpw_1TL)    + 
                            gp.quicksum(self.mc_bpw_TLM[i].Pi*val_bpw_TLM[i]    for i in self.mc_bpw_TLM)    +  
                            gp.quicksum(self.mc_bpw_M[i].Pi*val_bpw_M[i]        for i in self.mc_bpw_M)          ) +
                        (   gp.quicksum(self.mc_bps_0[i].Pi*val_bps_0[i]        for i in self.mc_bps_0)      + 
                            gp.quicksum(self.mc_bps_1TL[i].Pi*val_bps_1TL[i]    for i in self.mc_bps_1TL)    + 
                            gp.quicksum(self.mc_bps_TLM[i].Pi*val_bps_TLM[i]    for i in self.mc_bps_TLM)    +  
                            gp.quicksum(self.mc_bps_M[i].Pi*val_bps_M[i]        for i in self.mc_bps_M)      + 
                            gp.quicksum(self.mc_bps_T[i].Pi*val_bps_T[i]        for i in self.mc_bps_T)          ))
            so_cw =     gp.quicksum( self.cw[k] * self.sv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C )
            so_cs =     gp.quicksum( self.cs[k][t] * self.sv_ac_sc[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C)
            so_brsc =   gp.quicksum( (self.cs[k][tp-t]+self.cc[k]) * self.sv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp > t)
            so_grsc =   gp.quicksum( (self.cs[k][t-tp]-self.cc[k]) * self.sv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp < t)
            so_cv =     gp.quicksum( self.cv * self.sv_aux_uv[(t,p)] for  t in self.T for p in self.P)
            sc_cuu =    gp.quicksum( self.cuu * self.sv_aux_uu[t] for t in self.T )
            so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv + sc_cuu)
            # so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv)
            self.sub_prob.setObjective( -so_val, gp.GRB.MINIMIZE )

            # Solve Subproblem
            self.sub_prob.optimize()

            # Update Master
            sa = gp.Column()
            sa.addTerms(val_b0.getValue(),              self.mc_b0)
            
            [sa.addTerms(val_bul_co[i].getValue(),      self.mc_bul_co[i])   for i in self.mc_bul_co     ]    
            [sa.addTerms(val_bul_nco[i].getValue(),     self.mc_bul_nco[i])  for i in self.mc_bul_nco    ]

            [sa.addTerms(val_bpw_0[i].getValue(),       self.mc_bpw_0[i])    for i in self.mc_bpw_0      ]    
            [sa.addTerms(val_bpw_1TL[i].getValue(),     self.mc_bpw_1TL[i])  for i in self.mc_bpw_1TL    ]
            [sa.addTerms(val_bpw_TLM[i].getValue(),     self.mc_bpw_TLM[i])  for i in self.mc_bpw_TLM    ]    
            [sa.addTerms(val_bpw_M[i].getValue(),       self.mc_bpw_M[i])    for i in self.mc_bpw_M      ]

            [sa.addTerms(val_bps_0[i].getValue(),       self.mc_bps_0[i])    for i in self.mc_bps_0      ]    
            [sa.addTerms(val_bps_1TL[i].getValue(),     self.mc_bps_1TL[i])  for i in self.mc_bps_1TL    ]
            [sa.addTerms(val_bps_TLM[i].getValue(),     self.mc_bps_TLM[i])  for i in self.mc_bps_TLM    ]    
            [sa.addTerms(val_bps_M[i].getValue(),       self.mc_bps_M[i])    for i in self.mc_bps_M      ]
            [sa.addTerms(val_bps_T[i].getValue(),       self.mc_bps_T[i])    for i in self.mc_bps_T      ]  

            sa_var = self.master.addVar(vtype = gp.GRB.CONTINUOUS, name= f"sa_{self.iter}", column = sa)

            # Save objective for phase 2
            self.mo_p2.add(sa_var, so_cost.getValue())

            # End Condition
            if self.master.ObjVal <= 0: 
                self.master.optimize()
                break

            self.iter += 1


    ##### Phase 2 #####
    def solve_phase2(self):

        # Update Master Model
        self.master.remove(self.mv_b0)
        self.master.remove(self.mv_bul_co)
        self.master.remove(self.mv_bul_nco)
        self.master.remove(self.mv_bpw_0)
        self.master.remove(self.mv_bpw_1TL)
        self.master.remove(self.mv_bpw_TLM)
        self.master.remove(self.mv_bpw_M)
        self.master.remove(self.mv_bps_0)
        self.master.remove(self.mv_bps_1TL)
        self.master.remove(self.mv_bps_TLM)
        self.master.remove(self.mv_bps_M)
        self.master.remove(self.mv_bps_T)
        self.master.setObjective(self.mo_p2, gp.GRB.MINIMIZE)

        # Initiate Beta Approximation
        beta_approx = {'b0': 0, 'bul': {}, 'bpw': {}, 'bps': {}}
        for i in self.P: beta_approx['bul'][i] = 0
        for i in itertools.product(self.M,self.D,self.K,self.C): beta_approx['bpw'][i] = 0
        for i in itertools.product(self.T,self.M,self.D,self.K,self.C): beta_approx['bps'][i] = 0

        # Start up model
        to_import_betas = True
        if self.import_model:
            model_to_import = gp.read(self.import_model)
            model_to_import.optimize()
            vars_to_import = model_to_import.getVars()
            for var in vars_to_import:
                col_to_import = model_to_import.getCol(var)
                sa_var = self.master.addVar(vtype = gp.GRB.CONTINUOUS, name= f"sa_{self.iter}", Column = col_to_import, obj=var.obj)
                self.iter += 1


        # Solve Phase 2
        self.iter += 1
        close_count = 0
        count_same = 0
        objs = []
        while True:

            # Update beta in the algoritm
            for point in range(len(self.beta_fun)):
                if self.iter >= self.beta_fun[point][0]: beta_alp = self.beta_fun[point][1]

            # If model is imported generates initial betas completely
            if self.import_model and to_import_betas:
                beta_alp = 0
                to_import_betas = False
            
            # Solve Master
            if self.iter%500 == 0: self.master.write(f'{self.export_model}')
            self.master.optimize()

            # Update beta approximation
            beta_approx['b0'] = (beta_alp)*beta_approx['b0'] + (1-beta_alp) * self.mc_b0.Pi

            for i in self.mc_bul_co: beta_approx['bul'][i] = (beta_alp)*beta_approx['bul'][i] + (1-beta_alp) * self.mc_bul_co[i].Pi
            for i in self.mc_bul_nco: beta_approx['bul'][i] = (beta_alp)*beta_approx['bul'][i] + (1-beta_alp) * self.mc_bul_nco[i].Pi

            for i in self.mc_bpw_0: beta_approx['bpw'][i] = (beta_alp)*beta_approx['bpw'][i] + (1-beta_alp) * self.mc_bpw_0[i].Pi
            for i in self.mc_bpw_1TL: beta_approx['bpw'][i] = (beta_alp)*beta_approx['bpw'][i] + (1-beta_alp) * self.mc_bpw_1TL[i].Pi
            for i in self.mc_bpw_TLM: beta_approx['bpw'][i] = (beta_alp)*beta_approx['bpw'][i] + (1-beta_alp) * self.mc_bpw_TLM[i].Pi
            for i in self.mc_bpw_M: beta_approx['bpw'][i] = (beta_alp)*beta_approx['bpw'][i] + (1-beta_alp) * self.mc_bpw_M[i].Pi
            
            for i in self.mc_bps_0: beta_approx['bps'][i] = (beta_alp)*beta_approx['bps'][i] + (1-beta_alp) * self.mc_bps_0[i].Pi
            for i in self.mc_bps_1TL: beta_approx['bps'][i] = (beta_alp)*beta_approx['bps'][i] + (1-beta_alp) * self.mc_bps_1TL[i].Pi
            for i in self.mc_bps_TLM: beta_approx['bps'][i] = (beta_alp)*beta_approx['bps'][i] + (1-beta_alp) * self.mc_bps_TLM[i].Pi
            for i in self.mc_bps_M: beta_approx['bps'][i] = (beta_alp)*beta_approx['bps'][i] + (1-beta_alp) * self.mc_bps_M[i].Pi
            for i in self.mc_bps_T: beta_approx['bps'][i] = (beta_alp)*beta_approx['bps'][i] + (1-beta_alp) * self.mc_bps_T[i].Pi
            
            # Generate Value Equations
            val_b0 = gp.LinExpr(1-self.gam)

            val_bul_co = {p: gp.LinExpr(self.sv_st_ul[p] - self.gam*self.sv_aux_ulp[p])  for p in self.PCO }
            val_bul_nco = {p: gp.LinExpr(self.sv_st_ul[p])  for p in self.PNCO }

            val_bpw_0 = {(0,d,k,c): gp.LinExpr(self.sv_st_pw[(0,d,k,c)] - self.gam*self.pea[(d,k,c)])  for d in self.D for k in self.K for c in self.C}
            val_bpw_1TL = {(m,d,k,c): gp.LinExpr(self.sv_st_pw[(m,d,k,c)] - self.gam*self.sv_aux_pwp[(m-1,d,k,c)])  for m,d,k,c in self.mTLdkc}
            val_bpw_TLM = {}
            for m,d,k,c in self.TLMdkc:
                val_bpw_TLM[(m,d,k,c)] = gp.LinExpr( self.sv_st_pw[(m,d,k,c)] - self.gam*self.sv_aux_pwp[(m-1,d,k,c)] )
                if d != self.D[0]:  val_bpw_TLM[(m,d,k,c)] += -self.gam * self.sv_aux_pwt_d[(m-1, self.D[self.D.index(d)-1] ,k,c)]
                if k != self.K[0]:  val_bpw_TLM[(m,d,k,c)] += -self.gam * self.sv_aux_pwt_k[(m-1,d, self.K[self.K.index(k)-1] ,c)]
                if d != self.D[-1]: val_bpw_TLM[(m,d,k,c)] +=  self.gam * self.sv_aux_pwt_d[(m-1,d,k,c)]
                if k != self.K[-1]: val_bpw_TLM[(m,d,k,c)] +=  self.gam * self.sv_aux_pwt_k[(m-1,d,k,c)]
            val_bpw_M = {}
            for m,d,k,c in itertools.product( self.M[-1:], self.D, self.K, self.C):
                val_bpw_M[(m,d,k,c)] = gp.LinExpr( self.sv_st_pw[(m,d,k,c)] - self.gam*gp.quicksum( self.sv_aux_pwp[mm,d,k,c] for mm in self.M[-2:] ) )
                if d != self.D[0]:  val_bpw_M[(m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pwt_d[(mm, self.D[self.D.index(d)-1] ,k,c)] for mm in self.M[-2:])
                if k != self.K[0]:  val_bpw_M[(m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pwt_k[(mm,d, self.K[self.K.index(k)-1] ,c)] for mm in self.M[-2:])
                if d != self.D[-1]: val_bpw_M[(m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pwt_d[(mm,d,k,c)] for mm in self.M[-2:])
                if k != self.K[-1]: val_bpw_M[(m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pwt_k[(mm,d,k,c)] for mm in self.M[-2:])    

            val_bps_0 = {(t,0,d,k,c): gp.LinExpr(self.sv_st_ps[(t,0,d,k,c)])  for t in self.T[:-1] for d in self.D for k in self.K for c in self.C }
            val_bps_1TL = {(t,m,d,k,c): gp.LinExpr(self.sv_st_ps[(t,m,d,k,c)] - self.gam*self.sv_aux_psp[(t+1,m-1,d,k,c)]) for t,m,d,k,c in self.tmTLdkc}
            val_bps_TLM = {}
            for t,m,d,k,c in self.tTLMdkc:
                val_bps_TLM[(t,m,d,k,c)] = gp.LinExpr( self.sv_st_ps[(t,m,d,k,c)] - self.gam*self.sv_aux_psp[(t+1,m-1,d,k,c)] )
                if d != self.D[0]:  val_bps_TLM[(t,m,d,k,c)] += -self.gam * self.sv_aux_pst_d[(t+1,m-1, self.D[self.D.index(d)-1] ,k,c)]
                if k != self.K[0]:  val_bps_TLM[(t,m,d,k,c)] += -self.gam * self.sv_aux_pst_k[(t+1,m-1,d, self.K[self.K.index(k)-1] ,c)]
                if d != self.D[-1]: val_bps_TLM[(t,m,d,k,c)] +=  self.gam * self.sv_aux_pst_d[(t+1,m-1,d,k,c)]
                if k != self.K[-1]: val_bps_TLM[(t,m,d,k,c)] +=  self.gam * self.sv_aux_pst_k[(t+1,m-1,d,k,c)]
            val_bps_M = {}
            for t,m,d,k,c in itertools.product(self.T[:-1], self.M[-1:], self.D, self.K, self.C):
                val_bps_M[(t,m,d,k,c)] = gp.LinExpr( self.sv_st_ps[(t,m,d,k,c)] - self.gam*gp.quicksum( self.sv_aux_psp[(t+1,mm,d,k,c)] for mm in self.M[-2:] ) )
                if d != self.D[0]:  val_bps_M[(t,m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pst_d[(t+1,mm, self.D[self.D.index(d)-1] ,k,c)] for mm in self.M[-2:])
                if k != self.K[0]:  val_bps_M[(t,m,d,k,c)] += -self.gam * gp.quicksum( self.sv_aux_pst_k[(t+1,mm,d, self.K[self.K.index(k)-1] ,c)] for mm in self.M[-2:])
                if d != self.D[-1]: val_bps_M[(t,m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pst_d[(t+1,mm,d,k,c)] for mm in self.M[-2:])
                if k != self.K[-1]: val_bps_M[(t,m,d,k,c)] +=  self.gam * gp.quicksum( self.sv_aux_pst_k[(t+1,mm,d,k,c)] for mm in self.M[-2:])    
            val_bps_T = {(self.T[-1],m,d,k,c): gp.LinExpr( self.sv_st_ps[(self.T[-1],m,d,k,c)] )  for m in self.M for d in self.D for k in self.K for c in self.C}

            # Update Subproblem
            so_val =   ((   beta_approx['b0'] * val_b0                                                       ) +
                        (   gp.quicksum(beta_approx['bul'][i]*val_bul_co[i]        for i in self.mc_bul_co)     + 
                            gp.quicksum(beta_approx['bul'][i]*val_bul_nco[i]       for i in self.mc_bul_nco)        ) + 
                        (   gp.quicksum(beta_approx['bpw'][i]*val_bpw_0[i]         for i in self.mc_bpw_0)      + 
                            gp.quicksum(beta_approx['bpw'][i]*val_bpw_1TL[i]       for i in self.mc_bpw_1TL)    + 
                            gp.quicksum(beta_approx['bpw'][i]*val_bpw_TLM[i]       for i in self.mc_bpw_TLM)    +  
                            gp.quicksum(beta_approx['bpw'][i]*val_bpw_M[i]         for i in self.mc_bpw_M)          ) +
                        (   gp.quicksum(beta_approx['bps'][i]*val_bps_0[i]         for i in self.mc_bps_0)      + 
                            gp.quicksum(beta_approx['bps'][i]*val_bps_1TL[i]       for i in self.mc_bps_1TL)    + 
                            gp.quicksum(beta_approx['bps'][i]*val_bps_TLM[i]       for i in self.mc_bps_TLM)    +  
                            gp.quicksum(beta_approx['bps'][i]*val_bps_M[i]         for i in self.mc_bps_M)      + 
                            gp.quicksum(beta_approx['bps'][i]*val_bps_T[i]         for i in self.mc_bps_T)          ))
            so_cw =     gp.quicksum( self.cw[k] * self.sv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C )
            so_cs =     gp.quicksum( self.cs[k][t] * self.sv_ac_sc[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C)
            so_brsc =   gp.quicksum( (self.cs[k][tp-t]+self.cc[k]) * self.sv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp > t)
            so_grsc =   gp.quicksum( (self.cs[k][t-tp]-self.cc[k]) * self.sv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp < t)
            so_cv =     gp.quicksum( self.cv * self.sv_aux_uv[(t,p)] for  t in self.T for p in self.P)
            sc_cuu =    gp.quicksum( self.cuu * self.sv_aux_uu[t] for t in self.T )
            so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv + sc_cuu)
            # so_cost = (so_cw + so_cs + so_brsc - so_grsc + so_cv)

            self.sub_prob.setObjective( so_cost-so_val, gp.GRB.MINIMIZE )

            # Solve Subproblem
            # if self.iter%500 == 0: sub_prob.write(f'p2s.lp')
            self.sub_prob.optimize()
            if self.iter%250 == 0:
                print(f"PHASE 2 Sub Iter {self.iter}:\t\t{self.sub_prob.ObjVal}")


            # Update Master
            sa = gp.Column()
            sa.addTerms(val_b0.getValue(),              self.mc_b0)
            [sa.addTerms(val_bul_co[i].getValue(),      self.mc_bul_co[i])   for i in self.mc_bul_co     ]    
            [sa.addTerms(val_bul_nco[i].getValue(),     self.mc_bul_nco[i])  for i in self.mc_bul_nco    ]
            [sa.addTerms(val_bpw_0[i].getValue(),       self.mc_bpw_0[i])    for i in self.mc_bpw_0      ]    
            [sa.addTerms(val_bpw_1TL[i].getValue(),     self.mc_bpw_1TL[i])  for i in self.mc_bpw_1TL    ]
            [sa.addTerms(val_bpw_TLM[i].getValue(),     self.mc_bpw_TLM[i])  for i in self.mc_bpw_TLM    ]    
            [sa.addTerms(val_bpw_M[i].getValue(),       self.mc_bpw_M[i])    for i in self.mc_bpw_M      ]
            [sa.addTerms(val_bps_0[i].getValue(),       self.mc_bps_0[i])    for i in self.mc_bps_0      ]    
            [sa.addTerms(val_bps_1TL[i].getValue(),     self.mc_bps_1TL[i])  for i in self.mc_bps_1TL    ]
            [sa.addTerms(val_bps_TLM[i].getValue(),     self.mc_bps_TLM[i])  for i in self.mc_bps_TLM    ]    
            [sa.addTerms(val_bps_M[i].getValue(),       self.mc_bps_M[i])    for i in self.mc_bps_M      ]
            [sa.addTerms(val_bps_T[i].getValue(),       self.mc_bps_T[i])    for i in self.mc_bps_T      ]  
            self.sa_var = self.master.addVar(vtype = gp.GRB.CONTINUOUS, name= f"sa_{self.iter}", column = sa, obj=so_cost.getValue())


            # End Conditions
            if self.sub_prob.ObjVal >= 0:
                close_count += 1
            else:
                close_count = 0
            if close_count >= 1000:
                self.master.optimize()
                self.master.write(f'{self.export_model}')
                break
            if self.iter >= self.iter_lims:
                self.master.optimize()
                self.master.write(f'{self.export_model}')
                break
            if count_same >= 100:
                self.master.optimize()
                self.master.write(f'{self.export_model}')
                break
            
            objs.append(self.sub_prob.ObjVal)
            objs = objs[-2:]
            if len(objs) >= 2 and objs[-1] == objs[-2]: count_same += 1
            else: count_same = 0

            # Trim State - Actions
            # if self.iter%500 == 0:
            #     init_len = len(self.master.getVars())
            #     for i in self.master.getVars(): 
            #         if i.X == 0 and np.random.random() >= 0.8: 
            #             self.master.remove(i)
            #     final_len = len(self.master.getVars())
            #     print(f'PHASE 2 ITER {self.iter} Trimmed: \t{init_len - final_len}')

            self.iter += 1


    ##### Save Data #####
    def save_data(self):
        self.betas = {'b0': self.mc_b0.Pi, 'bul': {}, 'bpw': {}, 'bps': {}}

        for p in self.PCO: self.betas['bul'][p] = self.mc_bul_co[p].Pi 
        for p in self.PNCO: self.betas['bul'][p] = self.mc_bul_nco[p].Pi 

        for i in itertools.product(self.M[0:1], self.D, self.K, self.C): self.betas['bpw'][i] = self.mc_bpw_0[i].Pi
        for i in self.mTLdkc: self.betas['bpw'][i] = self.mc_bpw_1TL[i].Pi
        for i in self.TLMdkc: self.betas['bpw'][i] = self.mc_bpw_TLM[i].Pi
        for i in itertools.product(self.M[-1:], self.D, self.K, self.C): self.betas['bpw'][i] = self.mc_bpw_M[i].Pi

        for i in itertools.product(self.T[:-1], self.M[0:1], self.D, self.K, self.C): self.betas['bps'][i] = self.mc_bps_0[i].Pi
        for i in self.tmTLdkc: self.betas['bps'][i] = self.mc_bps_1TL[i].Pi
        for i in self.tTLMdkc: self.betas['bps'][i] = self.mc_bps_TLM[i].Pi
        for i in itertools.product(self.T[:-1], self.M[-1:], self.D, self.K, self.C): self.betas['bps'][i] = self.mc_bps_M[i].Pi
        for i in itertools.product(self.T[-1:], self.M, self.D, self.K, self.C): self.betas['bps'][i] = self.mc_bps_T[i].Pi

        with open(os.path.join(self.my_path, self.export_data), 'wb') as outp:
            pickle.dump(self.betas, outp, pickle.HIGHEST_PROTOCOL)