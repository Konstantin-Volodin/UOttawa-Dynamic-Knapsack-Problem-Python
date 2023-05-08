##### Initialization & Changeable Parameters #####
from modules import data_import
from gurobipy import *
import gurobipy as gp
import itertools
import os.path
import pickle
from copy import deepcopy
import numpy as np
from tqdm.auto import trange
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

class simulation_handler:
 
    ##### Initialize #####
    def __init__(self, sim_param, sim_paths):
        self.replications = sim_param['replications']
        self.warm_up = sim_param['warm_up']
        self.duration = sim_param['duration']
        
        self.import_data = sim_paths['import_params']
        self.import_beta = sim_paths['import_betas']

        self.export_txt = sim_paths['export_summary_costs']
        self.export_pic = sim_paths['export_summary_picture']

        self.export_state_my = sim_paths['export_state_myopic']
        self.export_state_md = sim_paths['export_state_mdp']
        self.export_cost_my = sim_paths['export_cost_myopic']
        self.export_cost_md = sim_paths['export_cost_mdp']
        self.export_util_my = sim_paths['export_util_myopic']
        self.export_util_md = sim_paths['export_util_mdp']
        self.export_sa_my = sim_paths['export_sa_myopic']
        self.export_sa_md = sim_paths['export_sa_mdp']
        self.my_path = os.getcwd()
        

    ##### Read Data #####
    def read_data(self):
        self.input_data = data_import.read_data(os.path.join(self.my_path, self.import_data))
        
        # self.export_state_my = os.path.join(self.my_path,self.export_state_my)
        # self.export_state_md = os.path.join(self.my_path,self.export_state_md)

        # Quick Assess to Various Parameters
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

        # Initial State
        self.init_state = {'ul': self.E_UL, 'pw': self.E_PW, 'ps': self.E_PS}
        for i in itertools.product(self.M,self.D,self.K,self.C): self.init_state['pw'][i] = 0
        for i in itertools.product(self.T,self.M,self.D,self.K,self.C): self.init_state['ps'][i] = 0
        for i in itertools.product(self.P): self.init_state['ul'][i] = 0

        # Betas
        with open(os.path.join(self.my_path, self.import_beta), 'rb') as handle:
            self.betas = pickle.load(handle)


    ##### Misc Function #####
    def retrive_surg_subset(self, df, d, k, c, m, day):
        '''
        ### Given a patient dataset, retrieves patient who are specific surgery type (accomodates transitions)
        Parameters: df - dataset, d - complexity, k - priority, c - surgery type, m - filter by wait time, day - current day
        '''
        # Filter on complexity, priority, and surgery type
        df_last_rows = df.groupby('id').tail(1).reset_index()
        df_surg_subset = df_last_rows.query(f"priority=='{k}' and complexity=='{d}' and surgery=='{c}'")
        df_surg_full_subset = df[df['id'].isin(df_surg_subset['id'])]

        # Filter on wait time
        if m != self.M[-1]:
            df_wait_subset = df_surg_full_subset.query(f"arrived_on == {day-m}")
            df_wat_full_subset = df[df['id'].isin(df_wait_subset['id'])]
        else:
            df_wait_subset = df_surg_full_subset.query(f"action == 'arrived'").query(f"arrived_on <= {day-m}")
            df_wat_full_subset = df[df['id'].isin(df_wait_subset['id'])]

        return(df_wat_full_subset)


    ##### Myopic Model #####
    def generate_myopic(self):
        self.myopic = gp.Model('Myopic')
        self.myopic.params.LogToConsole = 0

        # Cost Params
        self.myv_cost_cw = self.myopic.addVars(self.K, vtype=gp.GRB.CONTINUOUS, name='var_cost_cw')
        self.myv_cost_cs = self.myopic.addVars(self.K,[0]+self.T, vtype=gp.GRB.CONTINUOUS, name='var_cost_cs')
        self.myv_cost_cc = self.myopic.addVars(self.K, vtype=gp.GRB.CONTINUOUS, name='var_cost_cc')
        self.myv_cost_cv = self.myopic.addVar(vtype=gp.GRB.CONTINUOUS, name='var_cost_cv')
        self.myv_cost_cuu = self.myopic.addVar(vtype=gp.GRB.CONTINUOUS, name='var_cost_cuu')

        # Fix Costs
        for k in self.K: self.myv_cost_cw[k].UB = self.cw[k]; self.myv_cost_cw[k].LB = self.cw[k];
        for t,k in itertools.product(self.T, self.K): self.myv_cost_cs[(k,t)].UB = self.cs[k][t]; self.myv_cost_cs[(k,t)].LB = self.cs[k][t];
        for k in self.K: self.myv_cost_cc[k].UB = self.cc[k]; self.myv_cost_cc[k].LB = self.cc[k];
        for k in self.K: self.myv_cost_cv.UB = self.cv; self.myv_cost_cv.LB = self.cv;
        self.myv_cost_cuu.UB = self.cuu; self.myv_cost_cuu.LB = self.cuu; 

        # State Action & Auxiliary Variables
        self.myv_st_ul = self.myopic.addVars(self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_state_ul')
        self.myv_st_pw = self.myopic.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_pw')
        self.myv_st_ps = self.myopic.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_ps')

        self.myv_ac_sc = self.myopic.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_sc')
        self.myv_ac_rsc = self.myopic.addVars(self.T, self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_rsc')

        self.myv_aux_uv = self.myopic.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
        self.myv_aux_uvb = self.myopic.addVars(self.T, self.P, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

        self.myv_aux_ulp = self.myopic.addVars(self.PCO, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
        self.myv_aux_ulb = self.myopic.addVars(self.PCO, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

        self.myv_aux_uup = self.myopic.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
        self.myv_aux_pwp = self.myopic.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
        self.myv_aux_psp = self.myopic.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

        self.myv_aux_pwt_d = self.myopic.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
        self.myv_aux_pwt_k = self.myopic.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
        self.myv_aux_pst_d = self.myopic.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
        self.myv_aux_pst_k = self.myopic.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

        self.myv_aux_uu = self.myopic.addVars(self.T, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uu')
        self.myv_aux_uub = self.myopic.addVars(self.T, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_uub')

        # Definition of auxiliary variables
        self.myc_uup = self.myopic.addConstrs((self.myv_aux_uup[(t,p)] == gp.quicksum( self.U[(p,d,c)] * self.myv_aux_psp[(t,m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C) for t in self.T for p in self.P), name='con_auxiliary_uup')
        self.myc_pwp = self.myopic.addConstrs((self.myv_aux_pwp[(m,d,k,c)] == self.myv_st_pw[(m,d,k,c)] - gp.quicksum( self.myv_ac_sc[(t,m,d,k,c)] for t in self.T) for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwp')
        self.myc_psp = self.myopic.addConstrs((self.myv_aux_psp[(t,m,d,k,c)] == self.myv_st_ps[(t,m,d,k,c)] + self.myv_ac_sc[(t,m,d,k,c)] + gp.quicksum( self.myv_ac_rsc[tp,t,m,d,k,c] for tp in self.T) - gp.quicksum( self.myv_ac_rsc[t,tp,m,d,k,c] for tp in self.T) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pws')

        self.myc_aux_uv_0 = self.myopic.addConstrs((self.myv_aux_uv[(t,p)] >= 0 for t in self.T for p in self.P), name='con_auxiliary_uv_0')
        self.myc_aux_uv_0M = self.myopic.addConstrs((self.myv_aux_uv[(t,p)] <= self.BM * self.myv_aux_uvb[(t,p)] for t in self.T for p in self.P), name='con_auxiliary_uv_0M')
        self.myc_aux_uv_1 = self.myopic.addConstrs((self.myv_aux_uv[(1,p)] >= self.myv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.myv_st_ul[p] for p in self.P), name='con_auxiliary_uv_1')
        self.myc_aux_uv_1M = self.myopic.addConstrs((self.myv_aux_uv[(1,p)] <= (self.myv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.myv_st_ul[p]) + self.BM * (1 - self.myv_aux_uvb[(1, p)]) for p in self.P), name='con_auxiliary_uv_1M')
        self.myc_aux_uv_m = self.myopic.addConstrs((self.myv_aux_uv[(t, p)] >= (self.myv_aux_uup[(t, p)] - self.p_dat[p].expected_units) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_m')
        self.myc_aux_uv_mM = self.myopic.addConstrs((self.myv_aux_uv[(t, p)] <= (self.myv_aux_uup[(t,p)] - self.p_dat[p].expected_units) + self.BM * (1 - self.myv_aux_uvb[(t, p)]) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_mM')

        self.myc_aux_ulp_0 = self.myopic.addConstrs((self.myv_aux_ulp[p] >= 0 for p in self.PCO), name='con_auxiliary_ulp_0')
        self.myc_aux_ulp_0M = self.myopic.addConstrs((self.myv_aux_ulp[p] <= self.BM * self.myv_aux_ulb[p] for p in self.PCO), name='con_auxiliary_ulp_0M')
        self.myc_aux_ulp_p = self.myopic.addConstrs((self.myv_aux_ulp[p] >= (self.p_dat[p].expected_units + self.myv_st_ul[p] - self.myv_aux_uup[(1,p)]) for p in self.PCO), name='con_auxiliary_ulp_p')
        self.myc_aux_ulp_pM = self.myopic.addConstrs((self.myv_aux_ulp[p] <= (self.p_dat[p].expected_units + self.myv_st_ul[p] - self.myv_aux_uup[(1,p)]) + self.BM * (1-self.myv_aux_ulb[p]) for p in self.PCO), name='con_auxiliary_ulp_pM')

        self.myc_aux_pwt_d = self.myopic.addConstrs((self.myv_aux_pwt_d[(m,d,k,c)] == self.ptp_d[(d,c)] * self.myv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwp_d')
        self.myc_aux_pwt_k_0 = self.myopic.addConstrs((self.myv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_pwp[(m,d,k,c)] - self.myv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pwt_k_0')
        self.myc_aux_pwt_k_i = self.myopic.addConstrs((self.myv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_pwp[(m,d,k,c)] + self.myv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)] - self.myv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pwt_k')
        self.myc_aux_pwt_k_D = self.myopic.addConstrs((self.myv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_pwp[(m,d,k,c)] + self.myv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pwt_k_D')
        self.myc_aux_pwt_k = {**self.myc_aux_pwt_k_0, **self.myc_aux_pwt_k_i, **self.myc_aux_pwt_k_D}

        self.myc_aux_pst_d = self.myopic.addConstrs((self.myv_aux_pst_d[(t,m,d,k,c)] == self.ptp_d[(d,c)] * self.myv_aux_psp[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pst_d')
        self.myc_aux_pst_k_0 = self.myopic.addConstrs((self.myv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_psp[(t,m,d,k,c)] - self.myv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pst_k_0')
        self.myc_aux_pst_k_i = self.myopic.addConstrs((self.myv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_psp[(t,m,d,k,c)] + self.myv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)] - self.myv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pst_k')
        self.myc_aux_pst_k_D = self.myopic.addConstrs((self.myv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.myv_aux_psp[(t,m,d,k,c)] + self.myv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pst_k_D')
        self.myc_aux_pst_k = {**self.myc_aux_pst_k_0, **self.myc_aux_pst_k_i, **self.myc_aux_pst_k_D}

        self.myc_aux_uu_0 = self.myopic.addConstrs((self.myv_aux_uu[(t)] >= 0 for t in self.T), name='con_auxiliary_uu_0')
        self.myc_aux_uu_0M = self.myopic.addConstrs((self.myv_aux_uu[(t)] <= self.BM * self.myv_aux_uub[(t)] for t in self.T), name='con_auxiliary_uu_0M')
        self.myc_aux_uu_1 = self.myopic.addConstr((self.myv_aux_uu[(1)] >= self.p_dat[self.P[0]].expected_units + self.myv_st_ul[self.P[0]] - self.myv_aux_uup[1,self.P[0]]), name='con_auxiliary_uu_1')
        self.myc_aux_uu_1M = self.myopic.addConstr((self.myv_aux_uu[(1)] <= (self.p_dat[self.P[0]].expected_units + self.myv_st_ul[self.P[0]] - self.myv_aux_uup[1,self.P[0]]) + self.BM*(1 - self.myv_aux_uub[(1)])), name='con_auxiliary_uu_1M')
        self.myc_aux_uu_m = self.myopic.addConstrs((self.myv_aux_uu[(t)] >= (self.p_dat[self.P[0]].expected_units + self.myv_aux_uup[t,self.P[0]]) for t in self.T[1:]), name='con_auxiliary_uu_m')
        self.myc_aux_uu_mM = self.myopic.addConstrs((self.myv_aux_uu[(t)] <= (self.p_dat[self.P[0]].expected_units + self.myv_aux_uup[t,self.P[0]]) + self.BM * (1 - self.myv_aux_uub[(t)]) for t in self.T[1:]), name='con_auxiliary_uu_mM')

        # State Action Constraints
        self.myc_usage_1 = self.myopic.addConstrs((self.myv_aux_uup[(1,p)] <= self.p_dat[p].expected_units + self.myv_st_ul[p] + self.myv_aux_uv[(1,p)] for p in self.P), name='con_usage_1')
        self.myc_usage_tT = self.myopic.addConstrs((self.myv_aux_uup[(t,p)] <= self.p_dat[p].expected_units + self.myv_aux_uv[(t,p)] for t in self.T[1:] for p in self.P), name='con_usage_tT')

        self.myc_rescb = self.myopic.addConstrs((self.myv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T[1:] for tp in self.T[1:] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_reschedule_bounds')
        self.myc_rescb_ttp = self.myopic.addConstrs((self.myv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if t == tp == 1), name='con_reschedule_bounds_ttp')

        self.myc_resch = self.myopic.addConstrs((gp.quicksum(self.myv_ac_rsc[(t,tp,m,d,k,c)] for tp in self.T) <= self.myv_st_ps[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_resch')
        self.myc_sched = self.myopic.addConstrs((gp.quicksum(self.myv_ac_sc[(t,m,d,k,c)] for t in self.T) <= self.myv_st_pw[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_sched')

        # Objective Value
        self.myo_cw     =     gp.quicksum( self.myv_cost_cw[k] * self.myv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C )
        self.myo_cs     =     gp.quicksum( self.myv_cost_cs[(k,t)] * self.myv_ac_sc[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C)
        self.myo_brsc   =   gp.quicksum( (self.myv_cost_cs[(k,tp-t)]+self.myv_cost_cc[k]) * self.myv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp > t)
        self.myo_grsc   =   gp.quicksum( (self.myv_cost_cs[(k,t-tp)]-self.myv_cost_cc[k]) * self.myv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp < t)
        self.myo_cv     =     gp.quicksum( self.myv_cost_cv * self.myv_aux_uv[(t,p)] for  t in self.T for p in self.P)
        self.myo_cuu    =    gp.quicksum( self.cuu * self.myv_aux_uu[t] for t in self.T )
        self.myo_cost = (self.myo_cw + self.myo_cs + self.myo_brsc - self.myo_grsc + self.myo_cv + self.myo_cuu)
        self.myopic.setObjective( self.myo_cost, gp.GRB.MINIMIZE )


    ##### MDP Model #####
    def generate_mdp(self):
        self.MDP = gp.Model('MDP')
        self.MDP.params.LogToConsole = 0

        # State Action & Auxiliary Variables
        self.mdv_st_ul = self.MDP.addVars(self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_state_ul')
        self.mdv_st_pw = self.MDP.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_pw')
        self.mdv_st_ps = self.MDP.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb=0, name='var_state_ps')

        self.mdv_ac_sc = self.MDP.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_sc')
        self.mdv_ac_rsc = self.MDP.addVars(self.T, self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_action_rsc')

        self.mdv_aux_uv = self.MDP.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uv')
        self.mdv_aux_uvb = self.MDP.addVars(self.T, self.P, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_uvb')

        self.mdv_aux_ulp = self.MDP.addVars(self.PCO, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_ul')
        self.mdv_aux_ulb = self.MDP.addVars(self.PCO, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_ulb')

        self.mdv_aux_uup = self.MDP.addVars(self.T, self.P, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uup')
        self.mdv_aux_pwp = self.MDP.addVars(self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_pwp')
        self.mdv_aux_psp = self.MDP.addVars(self.T, self.M, self.D, self.K, self.C, vtype=gp.GRB.INTEGER, lb = 0, name='var_auxiliary_psp')

        self.mdv_aux_pwt_d = self.MDP.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_d')
        self.mdv_aux_pwt_k = self.MDP.addVars(self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pwt_k')
        self.mdv_aux_pst_d = self.MDP.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_d')
        self.mdv_aux_pst_k = self.MDP.addVars(self.T,self.M,self.D,self.K,self.C, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_pst_k')

        self.mdv_aux_uu = self.MDP.addVars(self.T, vtype=gp.GRB.CONTINUOUS, lb = 0, name='var_auxiliary_uu')
        self.mdv_aux_uub = self.MDP.addVars(self.T, vtype=gp.GRB.BINARY, lb = 0, name='var_auxiliary_uub')

        # Definition of auxiliary variables
        self.mdc_uup = self.MDP.addConstrs((self.mdv_aux_uup[(t,p)] == gp.quicksum( self.U[(p,d,c)] * self.mdv_aux_psp[(t,m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C) for t in self.T for p in self.P), name='con_auxiliary_uup')
        self.mdc_pwp = self.MDP.addConstrs((self.mdv_aux_pwp[(m,d,k,c)] == self.mdv_st_pw[(m,d,k,c)] - gp.quicksum( self.mdv_ac_sc[(t,m,d,k,c)] for t in self.T) for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwp')
        self.mdc_psp = self.MDP.addConstrs((self.mdv_aux_psp[(t,m,d,k,c)] == self.mdv_st_ps[(t,m,d,k,c)] + self.mdv_ac_sc[(t,m,d,k,c)] + gp.quicksum( self.mdv_ac_rsc[tp,t,m,d,k,c] for tp in self.T) - gp.quicksum( self.mdv_ac_rsc[t,tp,m,d,k,c] for tp in self.T) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pws')

        self.mdc_aux_uv_0 = self.MDP.addConstrs((self.mdv_aux_uv[(t,p)] >= 0 for t in self.T for p in self.P), name='con_auxiliary_uv_0')
        self.mdc_aux_uv_0M = self.MDP.addConstrs((self.mdv_aux_uv[(t,p)] <= self.BM * self.mdv_aux_uvb[(t,p)] for t in self.T for p in self.P), name='con_auxiliary_uv_0M')
        self.mdc_aux_uv_1 = self.MDP.addConstrs((self.mdv_aux_uv[(1,p)] >= self.mdv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.mdv_st_ul[p] for p in self.P), name='con_auxiliary_uv_1')
        self.mdc_aux_uv_1M = self.MDP.addConstrs((self.mdv_aux_uv[(1,p)] <= (self.mdv_aux_uup[(1, p)] - self.p_dat[p].expected_units - self.mdv_st_ul[p]) + self.BM * (1 - self.mdv_aux_uvb[(1, p)]) for p in self.P), name='con_auxiliary_uv_1M')
        self.mdc_aux_uv_m = self.MDP.addConstrs((self.mdv_aux_uv[(t, p)] >= (self.mdv_aux_uup[(t, p)] - self.p_dat[p].expected_units) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_m')
        self.mdc_aux_uv_mM = self.MDP.addConstrs((self.mdv_aux_uv[(t, p)] <= (self.mdv_aux_uup[(t,p)] - self.p_dat[p].expected_units) + self.BM * (1 - self.mdv_aux_uvb[(t, p)]) for t in self.T[1:] for p in self.P), name='con_auxiliary_uv_mM')

        self.mdc_aux_ulp_0 = self.MDP.addConstrs((self.mdv_aux_ulp[p] >= 0 for p in self.PCO), name='con_auxiliary_ulp_0')
        self.mdc_aux_ulp_0M = self.MDP.addConstrs((self.mdv_aux_ulp[p] <= self.BM * self.mdv_aux_ulb[p] for p in self.PCO), name='con_auxiliary_ulp_0M')
        self.mdc_aux_ulp_p = self.MDP.addConstrs((self.mdv_aux_ulp[p] >= (self.p_dat[p].expected_units + self.mdv_st_ul[p] - self.mdv_aux_uup[(1,p)]) for p in self.PCO), name='con_auxiliary_ulp_p')
        self.mdc_aux_ulp_pM = self.MDP.addConstrs((self.mdv_aux_ulp[p] <= (self.p_dat[p].expected_units + self.mdv_st_ul[p] - self.mdv_aux_uup[(1,p)]) + self.BM * (1-self.mdv_aux_ulb[p]) for p in self.PCO), name='con_auxiliary_ulp_pM')

        self.mdc_aux_pwt_d = self.MDP.addConstrs((self.mdv_aux_pwt_d[(m,d,k,c)] == self.ptp_d[(d,c)] * self.mdv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pwt_d')
        self.mdc_aux_pwt_k_0 = self.MDP.addConstrs((self.mdv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_pwp[(m,d,k,c)] - self.mdv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pwt_k_0')
        self.mdc_aux_pwt_k_i = self.MDP.addConstrs((self.mdv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_pwp[(m,d,k,c)] + self.mdv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)] - self.mdv_aux_pwt_d[(m,d,k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pwt_k')
        self.mdc_aux_pwt_k_D = self.MDP.addConstrs((self.mdv_aux_pwt_k[(m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_pwp[(m,d,k,c)] + self.mdv_aux_pwt_d[(m,self.D[self.D.index(d)-1],k,c)]) for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pwt_k_D')
        self.mdc_aux_pwt_k = {**self.mdc_aux_pwt_k_0, **self.mdc_aux_pwt_k_i, **self.mdc_aux_pwt_k_D}

        self.mdc_aux_pst_d = self.MDP.addConstrs((self.mdv_aux_pst_d[(t,m,d,k,c)] == self.ptp_d[(d,c)] * self.mdv_aux_psp[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_auxiliary_pst_d')
        self.mdc_aux_pst_k_0 = self.MDP.addConstrs((self.mdv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_psp[(t,m,d,k,c)] - self.mdv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[0]), name='con_auxiliary_pst_k_0')
        self.mdc_aux_pst_k_i = self.MDP.addConstrs((self.mdv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_psp[(t,m,d,k,c)] + self.mdv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)] - self.mdv_aux_pst_d[(t,m,d,k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d != self.D[0] and d != self.D[-1]), name='con_auxiliary_pst_k')
        self.mdc_aux_pst_k_D = self.MDP.addConstrs((self.mdv_aux_pst_k[(t,m,d,k,c)] == self.ptp_k[(k,c)] * (self.mdv_aux_psp[(t,m,d,k,c)] + self.mdv_aux_pst_d[(t,m,self.D[self.D.index(d)-1],k,c)]) for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C if d == self.D[-1]), name='con_auxiliary_pst_k_D')
        self.mdc_aux_pst_k = {**self.mdc_aux_pst_k_0, **self.mdc_aux_pst_k_i, **self.mdc_aux_pst_k_D}

        self.mdc_aux_uu_0 = self.MDP.addConstrs((self.mdv_aux_uu[(t)] >= 0 for t in self.T), name='con_auxiliary_uu_0')
        self.mdc_aux_uu_0M = self.MDP.addConstrs((self.mdv_aux_uu[(t)] <= self.BM * self.mdv_aux_uub[(t)] for t in self.T), name='con_auxiliary_uu_0M')
        self.mdc_aux_uu_1 = self.MDP.addConstr((self.mdv_aux_uu[(1)] >= self.p_dat[self.P[0]].expected_units + self.mdv_st_ul[self.P[0]] - self.mdv_aux_uup[1,self.P[0]]), name='con_auxiliary_uu_1')
        self.mdc_aux_uu_1M = self.MDP.addConstr((self.mdv_aux_uu[(1)] <= (self.p_dat[self.P[0]].expected_units + self.mdv_st_ul[self.P[0]] - self.mdv_aux_uup[1,self.P[0]]) + self.BM*(1 - self.mdv_aux_uub[(1)])), name='con_auxiliary_uu_1M')
        self.mdc_aux_uu_m = self.MDP.addConstrs((self.mdv_aux_uu[(t)] >= (self.p_dat[self.P[0]].expected_units + self.mdv_aux_uup[t,self.P[0]]) for t in self.T[1:]), name='con_auxiliary_uu_m')
        self.mdc_aux_uu_mM = self.MDP.addConstrs((self.mdv_aux_uu[(t)] <= (self.p_dat[self.P[0]].expected_units + self.mdv_aux_uup[t,self.P[0]]) + self.BM * (1 - self.mdv_aux_uub[(t)]) for t in self.T[1:]), name='con_auxiliary_uu_mM')

        # State Action Constraints
        self.mdc_usage_1 = self.MDP.addConstrs((self.mdv_aux_uup[(1,p)] <= self.p_dat[p].expected_units + self.mdv_st_ul[p] + self.mdv_aux_uv[(1,p)] for p in self.P), name='con_usage_1')
        self.mdc_usage_tT = self.MDP.addConstrs((self.mdv_aux_uup[(t,p)] <= self.p_dat[p].expected_units + self.mdv_aux_uv[(t,p)] for t in self.T[1:] for p in self.P), name='con_usage_tT')

        self.mdc_rescb = self.MDP.addConstrs((self.mdv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T[1:] for tp in self.T[1:] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_reschedule_bounds')
        self.mdc_rescb_ttp = self.MDP.addConstrs((self.mdv_ac_rsc[(t,tp,m,d,k,c)] == 0 for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if t == tp == 1), name='con_reschedule_bounds_ttp')

        self.mdc_resch = self.MDP.addConstrs((gp.quicksum(self.mdv_ac_rsc[(t,tp,m,d,k,c)] for tp in self.T) <= self.mdv_st_ps[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C), name='con_resch')
        self.mdc_sched = self.MDP.addConstrs((gp.quicksum(self.mdv_ac_sc[(t,m,d,k,c)] for t in self.T) <= self.mdv_st_pw[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C), name='con_sched')

        # Objective Value
        self.mdo_val_ul_co = gp.quicksum( self.betas['bul'][p] * self.mdv_aux_ulp[p] for p in self.PCO )

        self.mdo_val_pw_0 = gp.quicksum( self.betas['bpw'][(0,d,k,c)] * self.pea[(d,k,c)] for d in self.D for k in self.K for c in self.C )
        self.mdo_val_pw_1TL = gp.quicksum( self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwp[(m-1,d,k,c)] for m,d,k,c in self.mTLdkc )
        self.mdo_val_pw_TLM = (
            gp.quicksum( self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwp[(m-1,d,k,c)] for m,d,k,c in self.TLMdkc ) + 
            gp.quicksum( self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwt_d[(m-1, self.D[self.D.index(d)-1] ,k,c)] for m,d,k,c in self.TLMdkc if d != self.D[0] ) +
            gp.quicksum( self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwt_k[(m-1,d, self.K[self.K.index(k)-1] ,c)] for m,d,k,c in self.TLMdkc if k != self.K[0] ) +
            gp.quicksum( -self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwt_d[(m-1,d,k,c)] for m,d,k,c in self.TLMdkc if d != self.D[-1] ) +
            gp.quicksum( -self.betas['bpw'][(m,d,k,c)] * self.mdv_aux_pwt_k[(m-1,d,k,c)] for m,d,k,c in self.TLMdkc if k != self.K[-1] )
        )
        self.mdo_val_pw_M = (
            gp.quicksum( self.betas['bpw'][(self.M[-1],d,k,c)] * self.mdv_aux_pwp[(mm,d,k,c)] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C ) + 
            gp.quicksum( self.betas['bpw'][(self.M[-1],d,k,c)] * self.mdv_aux_pwt_d[(mm, self.D[self.D.index(d)-1] ,k,c)] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if d != self.D[0] ) +
            gp.quicksum( self.betas['bpw'][(self.M[-1],d,k,c)] * self.mdv_aux_pwt_k[(mm,d, self.K[self.K.index(k)-1] ,c)] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if k != self.K[0] ) +
            gp.quicksum( -self.betas['bpw'][(self.M[-1],d,k,c)] * self.mdv_aux_pwt_d[(mm,d,k,c)] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if d != self.D[-1] ) +
            gp.quicksum( -self.betas['bpw'][(self.M[-1],d,k,c)] * self.mdv_aux_pwt_k[(mm,d,k,c)] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if k != self.K[-1] )
        )

        self.mdo_val_ps_1TL = gp.quicksum( self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_psp[(t+1,m-1,d,k,c)] for t,m,d,k,c in self.tmTLdkc )
        self.mdo_val_ps_TLM = (
            gp.quicksum( self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_psp[(t+1,m-1,d,k,c)] for t,m,d,k,c in self.tTLMdkc ) + 
            gp.quicksum( self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_pst_d[(t+1,m-1, self.D[self.D.index(d)-1] ,k,c)] for t,m,d,k,c in self.tTLMdkc if d != self.D[0] ) +
            gp.quicksum( self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_pst_k[(t+1,m-1,d, self.K[self.K.index(k)-1] ,c)] for t,m,d,k,c in self.tTLMdkc if k != self.K[0] ) +
            gp.quicksum( -self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_pst_d[(t+1,m-1,d,k,c)] for t,m,d,k,c in self.tTLMdkc if d != self.D[-1] ) +
            gp.quicksum( -self.betas['bps'][(t,m,d,k,c)] * self.mdv_aux_pst_k[(t+1,m-1,d,k,c)] for t,m,d,k,c in self.tTLMdkc if k != self.K[-1] )
        )
        self.mdo_val_ps_M = (
            gp.quicksum( self.betas['bps'][(t,self.M[-1],d,k,c)] * self.mdv_aux_psp[(t+1,mm,d,k,c)] for mm in self.M[-2:] for t in self.T[:-1] for d in self.D for k in self.K for c in self.C ) + 
            gp.quicksum( self.betas['bps'][(t,self.M[-1],d,k,c)] * self.mdv_aux_pst_d[(t+1,mm, self.D[self.D.index(d)-1] ,k,c)] for t in self.T[:-1] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if d != self.D[0] ) +
            gp.quicksum( self.betas['bps'][(t,self.M[-1],d,k,c)] * self.mdv_aux_pst_k[(t+1,mm,d, self.K[self.K.index(k)-1] ,c)] for t in self.T[:-1] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if k != self.K[0] ) +
            gp.quicksum( -self.betas['bps'][(t,self.M[-1],d,k,c)] * self.mdv_aux_pst_d[(t+1,mm,d,k,c)] for t in self.T[:-1] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if d != self.D[-1] ) +
            gp.quicksum( -self.betas['bps'][(t,self.M[-1],d,k,c)] * self.mdv_aux_pst_k[(t+1,mm,d,k,c)] for t in self.T[:-1] for mm in self.M[-2:] for d in self.D for k in self.K for c in self.C if k != self.K[-1] )
        )
        self.mdo_val = (self.betas['b0'] + self.mdo_val_ul_co + (self.mdo_val_pw_0+self.mdo_val_pw_1TL+self.mdo_val_pw_TLM+self.mdo_val_pw_M) + (self.mdo_val_ps_1TL+self.mdo_val_ps_TLM+self.mdo_val_ps_M))

        self.mdo_cw =     gp.quicksum( self.cw[k] * self.mdv_aux_pwp[(m,d,k,c)] for m in self.M for d in self.D for k in self.K for c in self.C )
        self.mdo_cs =     gp.quicksum( self.cs[k][t] * self.mdv_ac_sc[(t,m,d,k,c)] for t in self.T for m in self.M for d in self.D for k in self.K for c in self.C)
        self.mdo_brsc =   gp.quicksum( (self.cs[k][tp-t]+self.cc[k]) * self.mdv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp > t)
        self.mdo_grsc =   gp.quicksum( (self.cs[k][t-tp]-self.cc[k]) * self.mdv_ac_rsc[(t,tp,m,d,k,c)] for t in self.T for tp in self.T for m in self.M for d in self.D for k in self.K for c in self.C if tp < t)
        self.mdo_cv =     gp.quicksum( self.cv * self.mdv_aux_uv[(t,p)] for  t in self.T for p in self.P)
        self.mdo_cuu =    gp.quicksum( self.cuu * self.mdv_aux_uu[t] for t in self.T )
        self.mdo_cost = (self.mdo_cw + self.mdo_cs + self.mdo_brsc - self.mdo_grsc + self.mdo_cv + self.mdo_cuu)

        self.MDP.setObjective( self.mdo_cost+(self.gam*self.mdo_val), gp.GRB.MINIMIZE )
    

    ##### Myopic Simulation
    def simulation_myopic(self):
    
        # Simulation Data
        self.my_sim_st = []
        self.my_sim_ac = []
        self.my_sim_cost = []
        self.my_sim_disc = []

        try: self.state_file = open(self.export_state_my,'x', newline="")
        except: self.state_file = open(self.export_state_my,'w', newline="")

        try: self.sa_file = open(self.export_sa_my,'x', newline="")
        except: self.sa_file = open(self.export_sa_my,'w', newline="")

        try: self.cost_file = open(self.export_cost_my,'x', newline="")
        except: self.cost_file = open(self.export_cost_my,'w', newline="")

        try: self.util_file = open(self.export_util_my,'x', newline="")
        except: self.util_file = open(self.export_util_my,'w', newline="")

        print(f"repl,period,policy,id,priority,complexity,surgery,action,arrived_on,sched_to,resch_from,resch_to,transition", file=self.state_file)
        print(f"repl,period,cost,cost_cw,cost_cs,cost_brsc,cost_grsc,cost_cv,cost_cuu", file=self.cost_file)
        print(f"repl,period,horizon_period,usage_admin,usage_OR", file=self.util_file)
        print(f"repl,period,state-aciton,value,t,tp,m,d,k,c,p,val", file=self.sa_file)

        # Simulation
        for repl in trange(self.replications, desc='Myopic'):

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
            state = deepcopy(self.init_state)

            # Detailed Logging Data
            patient_data_df = pd.DataFrame( columns=['repl', 'period', 'policy', 'id','priority', 'complexity', 'surgery','action','arrived_on','sched_to', 'resch_from', 'resch_to', 'transition'] )
            patient_id_count = 0

            # Single Replication
            for day in trange(self.duration, leave=False):

                # Init Patient Data
                patient_data = {'repl': [], 'period': [], 'policy': [], 'id': [], 'priority': [], 'complexity': [], 'surgery': [], 'action': [], 'arrived_on': [], 'sched_to': [], 'resch_from': [], 'resch_to': [], 'transition': []}
                if day == 0:
                    for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C):
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
                for k in self.K: self.myv_cost_cw[k].UB = self.cv-1; self.myv_cost_cw[k].LB = self.cv-1;
                for p in self.P: self.myv_st_ul[p].UB = state['ul'][p]; self.myv_st_ul[p].LB = state['ul'][p]
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): self.myv_st_pw[(m,d,k,c)].UB = round(state['pw'][(m,d,k,c)],0); self.myv_st_pw[(m,d,k,c)].LB = round(state['pw'][(m,d,k,c)],0)
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): self.myv_st_ps[(t,m,d,k,c)].UB = round(state['ps'][(t,m,d,k,c)],0); self.myv_st_ps[(t,m,d,k,c)].LB = round(state['ps'][(t,m,d,k,c)],0)
                self.myopic.optimize()

                # Save Cost (with normal logic)
                for k in self.K: self.myv_cost_cw[k].UB = self.cw[k]; self.myv_cost_cw[k].LB = self.cw[k];
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): self.myv_ac_sc[(t,m,d,k,c)].UB = round(self.myv_ac_sc[(t,m,d,k,c)].X,0); self.myv_ac_sc[(t,m,d,k,c)].LB = round(self.myv_ac_sc[(t,m,d,k,c)].X,0)
                for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): self.myv_ac_rsc[(t,tp,m,d,k,c)].UB = round(self.myv_ac_rsc[(t,tp,m,d,k,c)].X,0); self.myv_ac_rsc[(t,tp,m,d,k,c)].LB = round(self.myv_ac_rsc[(t,tp,m,d,k,c)].X,0)
                self.myopic.optimize()

                # Reset Myopic
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): self.myv_ac_sc[(t,m,d,k,c)].UB = gp.GRB.INFINITY; self.myv_ac_sc[(t,m,d,k,c)].LB = 0
                for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): self.myv_ac_rsc[(t,tp,m,d,k,c)].UB = gp.GRB.INFINITY; self.myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0

                rp_cost.append(self.myo_cost.getValue())
                print(f"{repl},{day}, {self.myo_cost.getValue()},{self.myo_cw.getValue()},{self.myo_cs.getValue()},{self.myo_brsc.getValue()},{self.myo_grsc.getValue()},{self.myo_cv.getValue()},{self.myo_cuu.getValue()}", file=self.cost_file)
                if day >= self.warm_up:
                    rp_disc = self.myo_cost.getValue() + self.gam*rp_disc

                # Save Action
                action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['sc'][i] = round(self.myv_ac_sc[i].X,0) 
                for i in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): action['rsc'][i] = round(self.myv_ac_rsc[i].X,0)
                for i in itertools.product(self.T,self.P): action['uv'][i] = self.myv_aux_uv[i].X
                for i in self.PCO: action['ulp'][i] = round(self.myv_aux_ulp[i].X,0)
                for i in itertools.product(self.T,self.P): action['uup'][i] = self.myv_aux_uup[i].X
                for i in itertools.product(self.M,self.D,self.K,self.C): action['pwp'][i] = round(self.myv_aux_pwp[i].X,0)
                for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['psp'][i] = round(self.myv_aux_psp[i].X,0)
                
                # Log Utilization
                for t in self.T:
                    print(f"{repl},{day},{t-1},{action['uup'][(t,self.P[0])]},{action['uup'][(t,self.P[1])]}", file=self.util_file)

                # Log State / Action
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): 
                    if state['pw'][(m,d,k,c)] != 0: print(f"{repl},{day},state,pw,t,tp,{m},{d},{k},{c},p,{state['pw'][(m,d,k,c)]}", file=self.sa_file)
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if state['ps'][(t,m,d,k,c)] != 0: print(f"{repl},{day},state,ps,{t},tp,{m},{d},{k},{c},p,{state['ps'][(t,m,d,k,c)]}", file=self.sa_file)

                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if action['sc'][(t,m,d,k,c)] != 0: print(f"{repl},{day},action,sc,{t},tp,{m},{d},{k},{c},p,{action['sc'][(t,m,d,k,c)]}", file=self.sa_file)
                for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): 
                    if action['rsc'][(t,tp,m,d,k,c)] != 0: print(f"{repl},{day},action,rsc,{t},{tp},{m},{d},{k},{c},p,{action['rsc'][(t,tp,m,d,k,c)]}", file=self.sa_file)
                
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): 
                    if action['pwp'][(m,d,k,c)] != 0: print(f"{repl},{day},post-state,pwp,t,tp,{m},{d},{k},{c},p,{action['pwp'][(m,d,k,c)]}", file=self.sa_file)
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if action['psp'][(t,m,d,k,c)] != 0: print(f"{repl},{day},post-state,psp,{t},tp,{m},{d},{k},{c},p,{action['psp'][(t,m,d,k,c)]}", file=self.sa_file)
                
                # Save Action Data for Logging
                # Add entries for schedulings
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):

                    skip = True
                    for t in self.T:
                        if action['sc'][(t,m,d,k,c)] > 0.001: skip = False
                    if skip == True: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)
                    
                    # Add entry for patients scheduled
                    sched = 0
                    for t in self.T:
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
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    
                    skip = True
                    for tp in self.T:
                        if action['rsc'][(t,tp,m,d,k,c)] >= 0.001: skip = False
                    if skip == True: continue

                    # Finds all scheduled patients based on t,m,d,k,c
                    pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Add entry for patient rescheduled
                    resched = 0
                    for tp in self.T:
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

                # Save Data to dataframe
                patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
                patient_data = {k : [] for k in patient_data}

                # Transition between States
                # Units Leftover / Unit Deviation
                for p in self.PCO: state['ul'][p] = action['ulp'][p] + round(my_strm_dev.uniform(self.p_dat[p].deviation[0],self.p_dat[p].deviation[1]), 2)
                for p in self.PNCO: state['ul'][p] = round(my_strm_dev.uniform(self.p_dat[p].deviation[0],self.p_dat[p].deviation[1]), 2)

                # Patients Waiting - set to post action
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): state['pw'][(m,d,k,c)] = action['pwp'][(m,d,k,c)]

                # Patients Waiting - calculate & execute D Transition
                my_ptw_d = {}
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        my_ptw_d[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], self.ptp_d[(d,c)] )

                # Save data on transitions
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        if my_ptw_d[(m,d,k,c)] == 0: continue
                        
                        # Find all unscheduled patients based on m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                        # Save Entry
                        for transition in range(my_ptw_d[(m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('myopic')
                            patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                            patient_data['priority'].append(k)
                            patient_data['complexity'].append(self.D[self.D.index(d)+1])
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
                
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): 
                    if d != self.D[-1] and m >= self.TL[c]: 
                        state['pw'][(m,self.D[self.D.index(d)+1],k,c)] = state['pw'][(m,self.D[self.D.index(d)+1],k,c)] + my_ptw_d[(m,d,k,c)]
                        state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_d[(m,d,k,c)]

                # Patients Waiting - calculate & execute K Transition
                my_ptw_k = {}
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        my_ptw_k[(m,d,k,c)] = my_strm_pwt.binomial(state['pw'][(m,d,k,c)], self.ptp_k[(k,c)] )

                # Save data on transitions
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        if my_ptw_k[(m,d,k,c)] == 0: continue
                        
                        # Find all unscheduled patients based on m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                        # Save Entry
                        for transition in range(my_ptw_k[(m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('myopic')
                            patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                            patient_data['priority'].append(self.K[self.K.index(k)+1])
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
                    
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): 
                    if k != self.K[-1] and m >= self.TL[c]: 
                        state['pw'][(m,d,self.K[self.K.index(k)+1],c)] = state['pw'][(m,d,self.K[self.K.index(k)+1],c)] + my_ptw_k[(m,d,k,c)]
                        state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - my_ptw_k[(m,d,k,c)]    

                # Patients Waiting - change wait time
                for d,k,c in itertools.product(self.D, self.K, self.C): state['pw'][(self.M[-1],d,k,c)] +=  state['pw'][(self.M[-2],d,k,c)]
                for m,d,k,c in itertools.product(self.M[1:-1][::-1], self.D, self.K, self.C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]

                # Patient Arrivals
                for d,k,c in itertools.product(self.D, self.K, self.C): 
                    arrivals = my_strm_arr.poisson(self.pea[(d,k,c)])

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

                # Patients Scheduled - post action
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): state['ps'][(t,m,d,k,c)] = action['psp'][(t,m,d,k,c)]

                # Patients Scheduled - calculate & execute D Transition
                my_pts_d = {}
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        my_pts_d[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], self.ptp_d[(d,c)] )

                # Save data on transitions
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        if my_pts_d[(t,m,d,k,c)] == 0: continue
                        
                        # Find all scheduled patients based on t,m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                        pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                        # Save Entry
                        for transition in range(my_pts_d[(t,m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('myopic')
                            patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                            patient_data['priority'].append(k)
                            patient_data['complexity'].append(self.D[self.D.index(d)+1])
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

                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): 
                    if d != self.D[-1] and m >= self.TL[c]: 
                        state['ps'][(t,m,self.D[self.D.index(d)+1],k,c)] = state['ps'][(t,m,self.D[self.D.index(d)+1],k,c)] + my_pts_d[(t,m,d,k,c)]
                        state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_d[(t,m,d,k,c)]

                # Patients Scheduled - calculate & execute K Transition
                my_pts_k = {}
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        my_pts_k[(t,m,d,k,c)] = my_strm_pst.binomial(state['ps'][(t,m,d,k,c)], self.ptp_k[(k,c)] )

                # Save data on transitions
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        if my_pts_k[(t,m,d,k,c)] == 0: continue
                        
                        # Find all scheduled patients based on t,m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                        pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                        # Save Entry
                        for transition in range(my_pts_k[(t,m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('myopic')
                            patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                            patient_data['priority'].append(self.K[self.K.index(k)+1])
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
                            
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): 
                    if k != self.K[-1] and m >= self.TL[c]: 
                        state['ps'][(t,m,d,self.K[self.K.index(k)+1],c)] = state['ps'][(t,m,d,self.K[self.K.index(k)+1],c)] + my_pts_k[(t,m,d,k,c)]
                        state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - my_pts_k[(t,m,d,k,c)]


                # Patients Scheduled  - change wait time
                for t,d,k,c in itertools.product(self.T, self.D, self.K, self.C): state['ps'][(t,self.M[-1],d,k,c)] +=  state['ps'][(t,self.M[-2],d,k,c)]
                for t,m,d,k,c in itertools.product(self.T, self.M[1:-1][::-1], self.D, self.K, self.C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
                for t,d,k,c in itertools.product(self.T, self.D, self.K, self.C): state['ps'][(t,0,d,k,c)] = 0
                
                # Patients Scheduled  - change scheduled time
                for t,m,d,k,c in itertools.product(self.T[:-1],self.M,self.D,self.K,self.C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): state['ps'][(self.T[-1],m,d,k,c)] = 0

                # Save Data to dataframe
                patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])

            # Save logging
            patient_data_df.to_csv(self.state_file, index=None, header=False)
                
            self.my_sim_st.append(rp_st)
            self.my_sim_ac.append(rp_ac)
            self.my_sim_cost.append(rp_cost)
            self.my_sim_disc.append(rp_disc)


    ##### MDP Simulation
    def simulation_mdp(self):

        # Simulation Data
        self.md_sim_st = []
        self.md_sim_ac = []
        self.md_sim_cost = []
        self.md_sim_disc = []

        try: self.state_file = open(self.export_state_md,'x', newline="")
        except: self.state_file = open(self.export_state_md,'w', newline="")

        try: self.sa_file = open(self.export_sa_md,'x', newline="")
        except: self.sa_file = open(self.export_sa_md,'w', newline="")

        try: self.cost_file = open(self.export_cost_md,'x', newline="")
        except: self.cost_file = open(self.export_cost_md,'w', newline="")

        try: self.util_file = open(self.export_util_md,'x', newline="")
        except: self.util_file = open(self.export_util_md,'w', newline="")

        print(f"repl,period,policy,id,priority,complexity,surgery,action,arrived_on,sched_to,resch_from,resch_to,transition", file=self.state_file)
        print(f"repl,period,cost,cost_cw,cost_cs,cost_brsc,cost_grsc,cost_cv,cost_cuu", file=self.cost_file)
        print(f"repl,period,horizon_period,usage_admin,usage_OR", file=self.util_file)
        print(f"repl,period,state-aciton,value,t,tp,m,d,k,c,p,val", file=self.sa_file)

        # Simulation
        for repl in trange(self.replications, desc='MDP'):
            
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
            state = deepcopy(self.init_state)
            
            # Detailed Logging Data
            patient_data_df = pd.DataFrame( columns=['repl', 'period', 'policy', 'id','priority', 'complexity', 'surgery','action','arrived_on','sched_to', 'resch_from', 'resch_to', 'transition'] )
            patient_id_count = 0

            for day in trange(self.duration, leave=False):

                # Init Patient Data
                patient_data = {'repl': [], 'period': [], 'policy': [], 'id': [], 'priority': [], 'complexity': [], 'surgery': [], 'action': [], 'arrived_on': [], 'sched_to': [], 'resch_from': [], 'resch_to': [], 'transition': []}
                if day == 0:
                    for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C):
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
                if day >= self.warm_up:
                    for p in self.P: self.mdv_st_ul[p].UB = state['ul'][p]; self.mdv_st_ul[p].LB = state['ul'][p]
                    for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): self.mdv_st_pw[(m,d,k,c)].UB = state['pw'][(m,d,k,c)]; self.mdv_st_pw[(m,d,k,c)].LB = state['pw'][(m,d,k,c)]
                    for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): self.mdv_st_ps[(t,m,d,k,c)].UB = state['ps'][(t,m,d,k,c)]; self.mdv_st_ps[(t,m,d,k,c)].LB = state['ps'][(t,m,d,k,c)]
                    self.MDP.optimize()
                else:

                    for k in self.K: self.myv_cost_cw[k].UB = self.cv-1; self.myv_cost_cw[k].LB = self.cv-1;
                    for p in self.P: self.myv_st_ul[p].UB = state['ul'][p]; self.myv_st_ul[p].LB = state['ul'][p]
                    for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): self.myv_st_pw[(m,d,k,c)].UB = round(state['pw'][(m,d,k,c)],0); self.myv_st_pw[(m,d,k,c)].LB = round(state['pw'][(m,d,k,c)],0)
                    for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): self.myv_st_ps[(t,m,d,k,c)].UB = round(state['ps'][(t,m,d,k,c)],0); self.myv_st_ps[(t,m,d,k,c)].LB = round(state['ps'][(t,m,d,k,c)],0)
                    self.myopic.optimize()

                # Save Cost
                if day >= self.warm_up:
                    rp_cost.append(self.mdo_cost.getValue())
                    print(f"{repl},{day}, {self.mdo_cost.getValue()},{self.mdo_cw.getValue()},{self.mdo_cs.getValue()},{self.mdo_brsc.getValue()},{self.mdo_grsc.getValue()},{self.mdo_cv.getValue()},{self.mdo_cuu.getValue()}", file=self.cost_file)
                    rp_disc = self.mdo_cost.getValue() + self.gam*rp_disc
                else:
                    for k in self.K: self.myv_cost_cw[k].UB = self.cw[k]; self.myv_cost_cw[k].LB = self.cw[k];
                    for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): self.myv_ac_sc[(t,m,d,k,c)].UB = round(self.myv_ac_sc[(t,m,d,k,c)].X,0); self.myv_ac_sc[(t,m,d,k,c)].LB = round(self.myv_ac_sc[(t,m,d,k,c)].X,0)
                    for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): self.myv_ac_rsc[(t,tp,m,d,k,c)].UB = round(self.myv_ac_rsc[(t,tp,m,d,k,c)].X,0); self.myv_ac_rsc[(t,tp,m,d,k,c)].LB = round(self.myv_ac_rsc[(t,tp,m,d,k,c)].X,0)
                    self.myopic.optimize()
                    rp_cost.append(self.myo_cost.getValue())
                    print(f"{repl},{day}, {self.myo_cost.getValue()},{self.myo_cw.getValue()},{self.myo_cs.getValue()},{self.myo_brsc.getValue()},{self.myo_grsc.getValue()},{self.myo_cv.getValue()},{self.myo_cuu.getValue()}", file=self.cost_file)

                # Save Action
                if day >= self.warm_up:
                    action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                    for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['sc'][i] = round(self.mdv_ac_sc[i].X,0)
                    for i in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): action['rsc'][i] = round(self.mdv_ac_rsc[i].X,0)
                    for i in itertools.product(self.T,self.P): action['uv'][i] = self.mdv_aux_uv[i].X
                    for i in self.PCO: action['ulp'][i] = round(self.mdv_aux_ulp[i].X,0)
                    for i in itertools.product(self.T,self.P): action['uup'][i] = self.mdv_aux_uup[i].X
                    for i in itertools.product(self.M,self.D,self.K,self.C): action['pwp'][i] = round(self.mdv_aux_pwp[i].X,0)
                    for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['psp'][i] = round(self.mdv_aux_psp[i].X,0)

                    for t in self.T:
                        print(f"{repl},{day},{t-1},{action['uup'][(t,self.P[0])]},{action['uup'][(t,self.P[1])]}", file=self.util_file)
                else:
                    
                    for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): self.myv_ac_sc[(t,m,d,k,c)].UB = gp.GRB.INFINITY; self.myv_ac_sc[(t,m,d,k,c)].LB = 0
                    for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): self.myv_ac_rsc[(t,tp,m,d,k,c)].UB = gp.GRB.INFINITY; self.myv_ac_rsc[(t,tp,m,d,k,c)].LB = 0
                    action = {'sc':{}, 'rsc':{}, 'uv': {}, 'ulp': {}, 'uup': {}, 'pwp':{}, 'psp': {}}
                    for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['sc'][i] = round(self.myv_ac_sc[i].X,0) 
                    for i in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): action['rsc'][i] = round(self.myv_ac_rsc[i].X,0)
                    for i in itertools.product(self.T,self.P): action['uv'][i] = self.myv_aux_uv[i].X
                    for i in self.PCO: action['ulp'][i] = round(self.myv_aux_ulp[i].X,0)
                    for i in itertools.product(self.T,self.P): action['uup'][i] = self.myv_aux_uup[i].X
                    for i in itertools.product(self.M,self.D,self.K,self.C): action['pwp'][i] = round(self.myv_aux_pwp[i].X,0)
                    for i in itertools.product(self.T,self.M,self.D,self.K,self.C): action['psp'][i] = round(self.myv_aux_psp[i].X,0)
                    
                    for t in self.T:
                        print(f"{repl},{day},{t-1},{action['uup'][(t,self.P[0])]},{action['uup'][(t,self.P[1])]}", file=self.util_file)

                # Log State / Action
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): 
                    if state['pw'][(m,d,k,c)] != 0: print(f"{repl},{day},state,pw,t,tp,{m},{d},{k},{c},p,{state['pw'][(m,d,k,c)]}", file=self.sa_file)
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if state['ps'][(t,m,d,k,c)] != 0: print(f"{repl},{day},state,ps,{t},tp,{m},{d},{k},{c},p,{state['ps'][(t,m,d,k,c)]}", file=self.sa_file)

                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if action['sc'][(t,m,d,k,c)] != 0: print(f"{repl},{day},action,sc,{t},tp,{m},{d},{k},{c},p,{action['sc'][(t,m,d,k,c)]}", file=self.sa_file)
                for t,tp,m,d,k,c in itertools.product(self.T,self.T,self.M,self.D,self.K,self.C): 
                    if action['rsc'][(t,tp,m,d,k,c)] != 0: print(f"{repl},{day},action,rsc,{t},{tp},{m},{d},{k},{c},p,{action['rsc'][(t,tp,m,d,k,c)]}", file=self.sa_file)
                
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): 
                    if action['pwp'][(m,d,k,c)] != 0: print(f"{repl},{day},post-state,pwp,t,tp,{m},{d},{k},{c},p,{action['pwp'][(m,d,k,c)]}", file=self.sa_file)
                for t,m,d,k,c in itertools.product(self.T,self.M,self.D,self.K,self.C): 
                    if action['psp'][(t,m,d,k,c)] != 0: print(f"{repl},{day},post-state,psp,{t},tp,{m},{d},{k},{c},p,{action['psp'][(t,m,d,k,c)]}", file=self.sa_file)

                # Save Action Data for Logging
                # Add entries for schedulings
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):

                    skip = True
                    for t in self.T:
                        if action['sc'][(t,m,d,k,c)] > 0.001: skip = False
                    if skip == True: continue
                    
                    # Find all unscheduled patients based on m,d,k,c
                    pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)
                    
                    # Add entry for patients scheduled
                    sched = 0
                    for t in self.T:
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
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    
                    skip = True
                    for tp in self.T: 
                        if action['rsc'][(t,tp,m,d,k,c)] >= 0.001: skip = False
                    if skip == True: continue

                    # Finds all scheduled patients based on t,m,d,k,c
                    pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                    pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                    pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                    # Add entry for patient rescheduled
                    resched = 0
                    for tp in self.T:
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
                
                ### Transition between States
                # Units Leftover / Unit Deviation   
                for p in self.PCO: state['ul'][p] = action['ulp'][p] + round(md_strm_dev.uniform(self.p_dat[p].deviation[0],self.p_dat[p].deviation[1]), 2)
                for p in self.PNCO: state['ul'][p] = round(md_strm_dev.uniform(self.p_dat[p].deviation[0],self.p_dat[p].deviation[1]), 2)

                # Patients Waiting - set to post action
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): state['pw'][(m,d,k,c)] = action['pwp'][(m,d,k,c)]

                # Patients Waiting - calculate & execute D Transition
                md_ptw_d = {}
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        md_ptw_d[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], self.ptp_d[(d,c)] )

                # Save data on transitions
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        if md_ptw_d[(m,d,k,c)] == 0: continue
                        
                        # Find all unscheduled patients based on m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                        # Save Entry
                        for transition in range(md_ptw_d[(m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('MDP')
                            patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                            patient_data['priority'].append(k)
                            patient_data['complexity'].append(self.D[self.D.index(d)+1])
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
                
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): 
                    if d != self.D[-1] and m >= self.TL[c]: 
                        state['pw'][(m,self.D[self.D.index(d)+1],k,c)] = state['pw'][(m,self.D[self.D.index(d)+1],k,c)] + md_ptw_d[(m,d,k,c)]
                        state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - md_ptw_d[(m,d,k,c)]

                # Patients Waiting - calculate & execute K Transition
                md_ptw_k = {}
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        md_ptw_k[(m,d,k,c)] = md_strm_pwt.binomial(state['pw'][(m,d,k,c)], self.ptp_k[(k,c)] )

                # Save data on transitions
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        if md_ptw_k[(m,d,k,c)] == 0: continue
                        
                        # Find all unscheduled patients based on m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_unsched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) == 1)

                        # Save Entry
                        for transition in range(md_ptw_k[(m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('MDP')
                            patient_data['id'].append(pat_unsched['id'].to_list()[transition])
                            patient_data['priority'].append(self.K[self.K.index(k)+1])
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
                    
                for m,d,k,c in itertools.product(self.M, self.D, self.K, self.C): 
                    if k != self.K[-1] and m >= self.TL[c]: 
                        state['pw'][(m,d,self.K[self.K.index(k)+1],c)] = state['pw'][(m,d,self.K[self.K.index(k)+1],c)] + md_ptw_k[(m,d,k,c)]
                        state['pw'][(m,d,k,c)] = state['pw'][(m,d,k,c)] - md_ptw_k[(m,d,k,c)]    

                # Patients Waiting - change wait time
                for d,k,c in itertools.product(self.D, self.K, self.C): state['pw'][(self.M[-1],d,k,c)] +=  state['pw'][(self.M[-2],d,k,c)]
                for m,d,k,c in itertools.product(self.M[1:-1][::-1], self.D, self.K, self.C): state['pw'][(m,d,k,c)] =  state['pw'][(m-1,d,k,c)]

                # Patient Arrivals
                for d,k,c in itertools.product(self.D, self.K, self.C): 
                    arrivals = md_strm_arr.poisson(self.pea[(d,k,c)])

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

                # Patients Scheduled - post action
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): state['ps'][(t,m,d,k,c)] = action['psp'][(t,m,d,k,c)]

                # Patients Scheduled - calculate & execute D Transition
                md_pts_d = {}
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        md_pts_d[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], self.ptp_d[(d,c)] )

                # Save data on transitions
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if d != self.D[-1] and m >= self.TL[c]: 
                        if md_pts_d[(t,m,d,k,c)] == 0: continue
                        
                        # Find all scheduled patients based on t,m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                        pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                        # Save Entry
                        for transition in range(md_pts_d[(t,m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('MDP')
                            patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                            patient_data['priority'].append(k)
                            patient_data['complexity'].append(self.D[self.D.index(d)+1])
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

                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): 
                    if d != self.D[-1] and m >= self.TL[c]: 
                        state['ps'][(t,m,self.D[self.D.index(d)+1],k,c)] = state['ps'][(t,m,self.D[self.D.index(d)+1],k,c)] + md_pts_d[(t,m,d,k,c)]
                        state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - md_pts_d[(t,m,d,k,c)]

                # Patients Scheduled - calculate & execute K Transition
                md_pts_k = {}
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        md_pts_k[(t,m,d,k,c)] = md_strm_pst.binomial(state['ps'][(t,m,d,k,c)], self.ptp_k[(k,c)] )

                # Save data on transitions
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C):
                    if k != self.K[-1] and m >= self.TL[c]: 
                        if md_pts_k[(t,m,d,k,c)] == 0: continue
                        
                        # Find all scheduled patients based on t,m,d,k,c
                        pat_subset = self.retrive_surg_subset(patient_data_df, d, k ,c, m, day)
                        pat_sched = pat_subset.query(f"action != 'transition'").groupby('id').filter(lambda x: len(x) >= 2).groupby('id').tail(1).reset_index()
                        pat_sched_subset = pat_sched.query(f"sched_to == {day+t-1} or resch_to == {day+t-1}")

                        # Save Entry
                        for transition in range(md_pts_k[(t,m,d,k,c)]):
                            patient_data['repl'].append(repl)
                            patient_data['period'].append(day)
                            patient_data['policy'].append('MDP')
                            patient_data['id'].append(pat_sched_subset['id'].to_list()[transition])
                            patient_data['priority'].append(self.K[self.K.index(k)+1])
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
                            
                for t,m,d,k,c in itertools.product(self.T, self.M, self.D, self.K, self.C): 
                    if k != self.K[-1] and m >= self.TL[c]: 
                        state['ps'][(t,m,d,self.K[self.K.index(k)+1],c)] = state['ps'][(t,m,d,self.K[self.K.index(k)+1],c)] + md_pts_k[(t,m,d,k,c)]
                        state['ps'][(t,m,d,k,c)] = state['ps'][(t,m,d,k,c)] - md_pts_k[(t,m,d,k,c)]

                # Patients Scheduled  - change wait time
                for t,d,k,c in itertools.product(self.T, self.D, self.K, self.C): state['ps'][(t,self.M[-1],d,k,c)] +=  state['ps'][(t,self.M[-2],d,k,c)]
                for t,m,d,k,c in itertools.product(self.T, self.M[1:-1][::-1], self.D, self.K, self.C): state['ps'][(t,m,d,k,c)] =  state['ps'][(t,m-1,d,k,c)]
                for t,d,k,c in itertools.product(self.T, self.D, self.K, self.C): state['ps'][(t,0,d,k,c)] = 0
                
                # Patients Scheduled  - change scheduled time
                for t,m,d,k,c in itertools.product(self.T[:-1],self.M,self.D,self.K,self.C): state['ps'][(t,m,d,k,c)] = state['ps'][(t+1,m,d,k,c)]
                for m,d,k,c in itertools.product(self.M,self.D,self.K,self.C): state['ps'][(self.T[-1],m,d,k,c)] = 0

                # Save Data to dataframe
                patient_data_df = pd.concat([patient_data_df, pd.DataFrame.from_dict(patient_data)])
            
            # Save logging
            patient_data_df.to_csv(self.state_file, index=None, header=False)

            self.md_sim_st.append(rp_st)
            self.md_sim_ac.append(rp_ac)
            self.md_sim_cost.append(rp_cost)
            self.md_sim_disc.append(rp_disc)


    ##### Reporting #####
    def save_data(self):
        self.my_avg_cost = np.average(np.transpose(self.my_sim_cost),axis=1)
        self.md_avg_cost = np.average(np.transpose(self.md_sim_cost),axis=1)
        x_dat = [i for i in range(len(self.my_avg_cost))]

        # Plot
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Myopic', 'MDP'))
        fig.add_trace( go.Line(x=x_dat, y=self.my_avg_cost), row=1, col=1 )
        fig.add_trace( go.Line(x=x_dat, y=self.md_avg_cost), row=1, col=2 )
        fig.write_html(os.path.join(self.my_path, self.export_pic))

        # Print Cost
        with open(os.path.join(self.my_path, self.export_txt), "w") as self.state_file:
            print(f'Myopic: \t Discounted Cost {np.average(self.my_sim_disc):.2f} \t Average Cost {np.average(self.my_avg_cost[self.warm_up:]):.2f} \t Warm Up Cost {np.average(self.my_avg_cost[:self.warm_up]):.2f}', file=self.state_file)
            print(f'MDP: \t\t Discounted Cost {np.average(self.md_sim_disc):.2f} \t Average Cost {np.average(self.md_avg_cost[self.warm_up:]):.2f} \t Warm Up Cost {np.average(self.md_avg_cost[:self.warm_up]):.2f}', file=self.state_file)