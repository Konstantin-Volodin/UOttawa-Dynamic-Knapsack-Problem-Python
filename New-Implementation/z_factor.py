#%%|
from copy import deepcopy
from Modules import data_import

import os.path
import pickle
import itertools

from gurobipy import *
import numpy as np
import pandas as pd
import openpyxl
import json

from pprint import pprint
import plotly.express as px

def return_data(import_beta, R3, Myopic):

    # Importing Data
    test_modifier = "cw1-cc5-cv10-gam99-"
    data_type = "smaller-full"

    import_data =  f"Data/sens-data/{data_type}/{test_modifier}{data_type}-nopri-data.xlsx"
    import_beta = f"Data/sens-data/{data_type}/betas/{test_modifier}{data_type}-nopri-optimal-R1R2.pkl"

    #region Model
    my_path = os.getcwd()
    input_data = data_import.read_data(os.path.join(my_path, import_data))

    # Quick Assess to Various Parameters
    TL = input_data.transition.wait_limit
    U = input_data.usage
    ptp_d = input_data.transition.transition_rate_comp
    ptp_k = input_data.transition.transition_rate_pri
    cw = input_data.model_param.cw
    cs = input_data.model_param.cs
    cv = input_data.model_param.cv

    ##### Generating Sets #####
    T = input_data.indices['t']
    M = input_data.indices['m']
    P = input_data.indices['p']
    D = input_data.indices['d']
    K = input_data.indices['k']
    C = input_data.indices['c']

    # Betas
    with open(os.path.join(my_path, import_beta), 'rb') as handle:
        betas = pickle.load(handle)

    # MDP
    MDP = Model('MDP')
    mdv_ac_sc = MDP.addVars(T, M, D, K, C, vtype=GRB.INTEGER, lb = 0, name='var_action_sc')
    mdo_cs = quicksum( cs[k][t] * mdv_ac_sc[(t,m,d,k,c)] for t in T for m in M for d in D for k in K for c in C)
    MDP.setObjective( mdo_cs,GRB.MINIMIZE )
    MDP.update()
    #endregion

    # Set Betas to 0
    if Myopic:
        betas['b0'] = 0
        for p in P:
            betas['bul'][p] = 0
        for m,d,k,c in itertools.product(M,D,K,C):
            betas['bpw'][(m,d,k,c)] = 0
        for t,m,d,k,c in itertools.product(T,M,D,K,C):
            betas['bps'][(t,m,d,k,c)] = 0

    # FACTORIZING
    sc_coef= {}
    for i in itertools.product(T,M,D,K,C):
        sc_coef[i] = 0
    # Cost Function
    for i in itertools.product(T,M,D,K,C):
        sc_coef[i] += mdv_ac_sc[i].Obj
    # Cost of Waiting
    for t,m,d,k,c in itertools.product(T,M,D,K,C):
        sc_coef[(t,m,d,k,c)] -= cw[k]    
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
    # Cost of under utilization
    if R3:
        for t,m,d,k,c in itertools.product(T,M,D,K,C):
            sc_coef[(t,m,d,k,c)] -= 1000 * U[('Admissions',d,c)]

    # Cost of over utilization
    sc_coef_prio = deepcopy(sc_coef)
    for t,m,d,k,c in itertools.product(T,M,D,K,C):
        for p in P:
            sc_coef_prio[(t,m,d,k,c)] += cv * U[(p,d,c)]

    coef_df = pd.Series(sc_coef).reset_index()
    coef_df.columns = ['T','M','D','K','C','Val']
    coef_df = coef_df.assign( C = lambda df: df['C'].map(lambda c: f"S{c.split('.')[0]}") )
    coef_df = coef_df.assign( D = lambda df: df['D'].map(lambda d: f"C{d[-1]}") )
    coef_df['DK'] = coef_df['D'] + "," + coef_df['K']
    coef_df['DKC'] = coef_df['D'] + "," + coef_df['K'] + "," + coef_df['C']

    coef_df_prio = pd.Series(sc_coef_prio).reset_index()
    coef_df_prio.columns = ['T','M','D','K','C','Val']
    # temp = list(map(
    #     lambda x: {x[0]: x[1] / input_data.ppe_data[x[0][0]].expected_units}, 
    #     input_data.usage.items()
    # ))
    # ppe_usage = {}
    # for i in temp: ppe_usage[list(i.keys())[0]] = list(i.values())[0]
    resource_usage = {}
    for d,c in itertools.product(D,C): resource_usage[(d,c)] = input_data.usage[('OR_Time'), d, c]
    resource_usages_list = list(map(lambda x: resource_usage[(x['D'],x['C'])], coef_df_prio[['D','C']].to_dict('records')))

    # coef_df_prio = coef_df_prio.assign( cw = lambda df: df['K'].map(lambda k: cw[k]) )
    # coef_df_prio = coef_df_prio.assign( or_usage = resource_usages_list )
    # coef_df_prio = coef_df_prio.assign( Val = coef_df_prio['Val']/coef_df_prio['or_usage'])
    coef_df_prio = coef_df_prio.assign( C = lambda df: df['C'].map(lambda c: f"S{c.split('.')[0]}") )
    coef_df_prio = coef_df_prio.assign( D = lambda df: df['D'].map(lambda d: f"C{d[-1]}") )
    coef_df_prio['DK'] = coef_df_prio['D'] + "," + coef_df_prio['K']
    coef_df_prio['DKC'] = coef_df_prio['D'] + "," + coef_df_prio['K'] + "," + coef_df['C']

    # BASIC DATA
    if not R3: 
        if not Myopic:
            coef_df.to_csv('Report_Stuff/R2_basic.csv', index=False)
        else:
            coef_df.to_csv('Report_Stuff/R2_basic_myopic.csv', index=False)
    else:
        if not Myopic:
            coef_df.to_csv('Report_Stuff/R3_basic.csv', index=False)
        else:
            coef_df.to_csv('Report_Stuff/R3_basic_myopic.csv', index=False)

    # PRIORITY DATA
    if not R3: 
        if not Myopic:
            coef_df_prio.to_csv('Report_Stuff/R2_prio.csv', index=False)
        else:
            coef_df_prio.to_csv('Report_Stuff/R2_prio_myopic.csv', index=False)
    else:
        if not Myopic:
            coef_df_prio.to_csv('Report_Stuff/R3_prio.csv', index=False)
        else:
            coef_df_prio.to_csv('Report_Stuff/R3_prio_myopic.csv', index=False)

    # Basic FIGURE
    # fig_b = px.line(
    #     coef_df.query(f"M == {M[-1]}"), x='T',y='Val',color='C',
    #     line_dash='DK',symbol = 'DK', title="Scheduling Horizon")
    # if not R3:
    #     fig_b.write_html("Report_Stuff/R2_basic.html")
    # else:
    #     fig_b.write_html("Report_Stuff/R3_basic.html")

    # # Priority FIGURE

    # fig_b = px.line(
    #     coef_df_prio.query(f"M == {M[-1]}"), x='T',y='Val',color='C',
    #     line_dash='DK',symbol = 'DK', title="Scheduling Priority")
    # if not R3:
    #     fig_b.write_html("Report_Stuff/R2_prio.html")
    # else:
    #     fig_b.write_html("Report_Stuff/R3_prio.html")

    # # Detailed Daily figures
    # for t in T:
    #     fig_d = px.line(
    #         coef_df.query(f"M == {M[-1]}"), x='T',y='Val',color='C',
    #         line_dash='DK',symbol = 'DK', title=f"Scheduling Objective - Week {t}"
    #     )
    #     temp_coef_df = coef_df.query(f"M == {M[-1]} and T == {t}")
    #     fig_d.update_xaxes(range=[t*0.9, t*1.1])
    #     fig_d.update_yaxes(range=[temp_coef_df.Val.min()-0.5, temp_coef_df.Val.max()+0.5])
    #     fig_d.show(renderer='browser')

    # # Waitlist differences
    # coef_df_m = coef_df.assign(
    #     MN = lambda val: val['M'].apply(
    #         lambda valy: "New Arr" if valy == 0 else 
    #         ("Waitlist - no TR" if valy <= 2 else "Waitlist - TR")
    #     )
    # )
    # for c in ['Surgery 1', 'Surgery 4', 'Surgery 6']:
    #     fig_m = px.line(
    #         coef_df_m.query(f"C == '{c}'"), y='Val', x='T', color='MN', 
    #         line_dash='DK',symbol = 'DK',
    #         title=f"Scheduling Objective - {c}"
    #     )
    #     temp_coef_df = coef_df.query(f"C == '{c}' and T == 1")
    #     fig_m.update_xaxes(range=[0.9, 1.1])
    #     fig_m.update_yaxes(range=[temp_coef_df.Val.min()-0.5, temp_coef_df.Val.max()+0.5])
    #     fig_m.show(renderer='browser')
# %%
