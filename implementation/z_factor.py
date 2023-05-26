#%% 
import pandas as pd
from modules import data_import

import os.path
import pickle
import itertools

from plotly.subplots import make_subplots
import plotly.graph_objects as go


#%%
def generate_z_score(path, modifier, myopic, cuu):
    # path = '/Users/konstantin/Documents/Projects/uOttawa/UOttawa-Dynamic-Knapsack-Problem-Python/implementation/data/full-sm'
    # modifier = '0-1'
    # myopic = True
    # cuu = False

    ##### READ DATA #####
    # IMPORT    
    input_path = os.path.join(path, 'input')
    output_path = os.path.join(path, 'res')
    cuu_val = float(modifier.split('-')[1])
    import_data = os.path.join(input_path, 'full-sm-np-dt.xlsx')
    import_beta = os.path.join(input_path, 'betas', f'full-sm-np-opt-{modifier}.pkl')
    input_data = data_import.read_data(os.path.join(import_data))

    # QUICK ACCESS
    TL = input_data.transition.wait_limit
    U = input_data.usage
    ptp_d = input_data.transition.transition_rate_comp
    ptp_k = input_data.transition.transition_rate_pri
    cw = input_data.model_param.cw
    cs = input_data.model_param.cs
    cv = input_data.model_param.cv

    # SETS
    T = input_data.indices['t']
    M = input_data.indices['m']
    P = input_data.indices['p']
    D = input_data.indices['d']
    K = input_data.indices['k']
    C = input_data.indices['c']


    ##### BETAS #####
    with open(os.path.join(import_beta), 'rb') as handle:
        betas = pickle.load(handle)

    # HANDLE MYOPIC
    if myopic:
        betas['b0'] = 0
        for p in P:
            betas['bul'][p] = 0
        for m,d,k,c in itertools.product(M,D,K,C):
            betas['bpw'][(m,d,k,c)] = 0
        for t,m,d,k,c in itertools.product(T,M,D,K,C):
            betas['bps'][(t,m,d,k,c)] = 0
        

    ##### FACTORIZING #####
    sc_coef= {}
    for i in itertools.product(T,M,D,K,C):
        sc_coef[i] = 0

    # UL
    for p,m,d,k,c in itertools.product(P, M, D, K, C):
        sc_coef[(1,m,d,k,c)] += betas['bul'][p]

    # PW
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

    # PS
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


    ##### COSTS #####
    # Cost of Scheduling
    for t, m, d, k, c in itertools.product(T,M,D,K,C):
        sc_coef[(t,m,d,k,c)] += cs[k][t]

    # Cost of Waiting (Additional)
    for t,m,d,k,c in itertools.product(T,M,D,K,C):
        if myopic:
            sc_coef[(t,m,d,k,c)] -= cv-1
        else:
            sc_coef[(t,m,d,k,c)] -= cw[k] 

    # Cost of under utilization
    if cuu:
        for t,m,d,k,c in itertools.product(T,M,D,K,C):
            sc_coef[(t,m,d,k,c)] -= 1000 * U[('Admissions',d,c)]
        
    # # Cost of over utilization
    # sc_coef = deepcopy(sc_coef)
    # for t,m,d,k,c in itertools.product(T,M,D,K,C):
    #     for p in P:
    #         sc_coef[(t,m,d,k,c)] += cv * U[(p,d,c)]


    ##### CLEANUP DATA #####
    # UNSCALED DATA
    coef_df = pd.Series(sc_coef).reset_index()
    coef_df.columns = ['T','M','D','K','C','Val']

    # SCALED DATA
    ppe_usage = {}
    for d, c in itertools.product(D, C):
        ppe_usage[(d,c)] = 0
        for p in P:
            if p == 'OR_Time': 
                ppe_usage[(d,c)] += input_data.usage[(p,d,c)] / (input_data.ppe_data[p].expected_units * cuu_val)
            else: 
                ppe_usage[(d,c)] += input_data.usage[(p,d,c)] / (input_data.ppe_data[p].expected_units)
    ppe_usage_list = list(map(lambda x: ppe_usage[(x['D'],x['C'])], coef_df[['D','C']].to_dict('records')))
    coef_df['Usage'] = ppe_usage_list
    coef_df['Val_Adj'] = coef_df['Val'] / coef_df['Usage']

    # CLEANING
    coef_df = coef_df.assign( C = lambda df: df['C'].map(lambda c: f"S{c.split('.')[0]}") )
    coef_df = coef_df.assign( D = lambda df: df['D'].map(lambda d: f"C{d[-1]}") )


    ##### EXPORT #####
    if myopic:
        coef_df.to_csv(os.path.join(output_path,'z_fact', f'full-sm-res-my-{modifier}.csv'), index=False)
    else:
        coef_df.to_csv(os.path.join(output_path,'z_fact', f'full-sm-res-mdp-{modifier}.csv'), index=False)

    # # VISUALIZED
    # fig = make_subplots(rows=2, cols=2,
    #                     subplot_titles=("C1 - Non Adjusted", "C1 - Adjusted", 
    #                                     "C2 - Non Adjusted", "C2 - Adjusted"))
    # colors = {'S1': 'rgb(27,158,119)', 'S4': 'rgb(217,95,2)', 'S6': 'rgb(117,112,179)'}

    # for c in coef_df['C'].drop_duplicates().tolist():
    #     fig.add_trace(go.Line(x=coef_df.query(f'M == 0 and D == "C1" and C == "{c}"')['T'], 
    #                           y=coef_df.query(f'M == 0 and D == "C1" and C == "{c}"')['Val'],
    #                           mode='lines+markers', line_color=colors[c],
    #                           legendgroup=f'group_{c}',name=f'Surg: {c}',),
    #                 row=1, col=1)
    #     fig.add_trace(go.Line(x=coef_df.query(f'M == 0 and D == "C1" and C == "{c}"')['T'], 
    #                           y=coef_df.query(f'M == 0 and D == "C1" and C == "{c}"')['Val_Adj'],
    #                           mode='lines+markers', line_color=colors[c],
    #                           legendgroup=f'group_{c}',name=f'Surg: {c}', showlegend=False ),
    #                   row=1, col=2)
    #     fig.add_trace(go.Line(x=coef_df.query(f'M == 0 and D == "C2" and C == "{c}"')['T'], 
    #                           y=coef_df.query(f'M == 0 and D == "C2" and C == "{c}"')['Val'],
    #                           mode='lines+markers', line_color=colors[c],
    #                           legendgroup=f'group_{c}',name=f'Surg: {c}', showlegend=False ),
    #                 row=2, col=1)
    #     fig.add_trace(go.Line(x=coef_df.query(f'M == 0 and D == "C2" and C == "{c}"')['T'], 
    #                           y=coef_df.query(f'M == 0 and D == "C2" and C == "{c}"')['Val_Adj'],
    #                           mode='lines+markers', line_color=colors[c],
    #                           legendgroup=f'group_{c}',name=f'Surg: {c}', showlegend=False ),
    #                   row=2, col=2)
    # fig.show(renderer='browser')


#%%
path = "F:/Documents/Projects/uOttawa/uOttawa-Dynamic_Knapsack_Problem-Python/implementation/data/full-sm"

modifs = ['0-1', '0-1.1', '0-1.2', '0-1.3',
          '1000-1', '1000-1.1', '1000-1.2', '1000-1.3']

for md in modifs:
    cuu_val = int(md.split('-')[0])
    if cuu_val == 1000:
        generate_z_score(path, md, False, True)
        generate_z_score(path, md, True, True)
    else:
        generate_z_score(path, md, False, False)
        generate_z_score(path, md, True, False)
# %%
