\ Model Master problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  28.5 sa_1 + 1.9 sa_2 + 28.5 sa_3 + 28.5 sa_4 + 28.5 sa_5 + 28.5 sa_6
   + 28.5 sa_7 - 25.575 sa_8 - 25.575 sa_9 + 1.9 sa_10 + 15.2 sa_11
   + 15.2 sa_12 - 10.035 sa_13 - 25.575 sa_14 + 28.5 sa_15
   + 17.290125 sa_16 - 6.795 sa_17 - 3.29725 sa_18
   - 4.154624999999999 sa_19 - 51.29624999999999 sa_20 - 46.166625 sa_21
   - 51.29624999999999 sa_22 - 25.575 sa_23 - 51.29624999999999 sa_24
   - 6.795 sa_25 - 9.294 sa_26 - 51.29624999999999 sa_27
   - 51.29624999999999 sa_28
Subject To
 con_beta_0: 0.05 sa_1 + 0.05 sa_2 + 0.05 sa_3 + 0.05 sa_4 + 0.05 sa_5
   + 0.05 sa_6 + 0.05 sa_7 + 0.05 sa_8 + 0.05 sa_9 + 0.05 sa_10
   + 0.05 sa_11 + 0.05 sa_12 + 0.05 sa_13 + 0.05 sa_14 + 0.05 sa_15
   + 0.05 sa_16 + 0.05 sa_17 + 0.05 sa_18 + 0.05 sa_19 + 0.05 sa_20
   + 0.05 sa_21 + 0.05 sa_22 + 0.05 sa_23 + 0.05 sa_24 + 0.05 sa_25
   + 0.05 sa_26 + 0.05 sa_27 + 0.05 sa_28 + 0.05 sa_30 + 0.05 sa_31
   + 0.05 sa_32 + 0.05 sa_33 + 0.05 sa_34 + 0.05 sa_35 + 0.05 sa_36
   + 0.05 sa_1156 + 0.05 sa_1157 = 1
 con_beta_ul_nco[PPE1]: 60 sa_1 + 60 sa_2 + 60 sa_3 + 60 sa_4 + 60 sa_5
   + 60 sa_6 + 60 sa_7 + 60 sa_8 + 60 sa_9 + 60 sa_10 + 60 sa_11 + 60 sa_12
   + 60 sa_13 + 60 sa_14 + 60 sa_15 + 60 sa_16 + 60 sa_17 + 60 sa_18
   + 60 sa_19 + 60 sa_20 + 60 sa_21 + 60 sa_22 + 60 sa_23 + 60 sa_24
   + 60 sa_25 + 60 sa_26 + 60 sa_27 + 60 sa_28 + 60 sa_30 + 60 sa_31
   + 60 sa_32 + 60 sa_33 + 60 sa_34 + 60 sa_35 + 60 sa_36 + 60 sa_1156
   + 60 sa_1157 >= 0
 con_beta_pw_0dkc[0,C1,P1,CPU1]: 21.45 sa_1 - 8.55 sa_2 - 8.55 sa_3
   - 8.55 sa_4 - 8.55 sa_5 - 8.55 sa_6 - 8.55 sa_7 - 8.55 sa_8 - 8.55 sa_9
   - 8.55 sa_10 + 5.45 sa_11 + 5.45 sa_12 - 8.55 sa_13 - 8.55 sa_14
   + 21.45 sa_15 - 8.55 sa_16 - 8.55 sa_17 - 8.55 sa_18 - 8.55 sa_19
   - 8.55 sa_20 - 8.55 sa_21 - 8.55 sa_22 - 8.55 sa_23 - 8.55 sa_24
   - 8.55 sa_25 - 8.55 sa_26 - 8.55 sa_27 - 8.55 sa_28 - 8.55 sa_30
   - 8.55 sa_31 - 8.55 sa_32 - 8.55 sa_33 - 8.55 sa_34 - 8.55 sa_35
   - 8.55 sa_36 - 8.55 sa_1156 - 8.55 sa_1157 >= 10
 con_beta_pw_0dkc[0,C2,P1,CPU1]: >= 0
 con_beta_pw_1TLdkc[1,C1,P1,CPU1]: 30 sa_7 >= 5
 con_beta_pw_1TLdkc[1,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[2,C1,P1,CPU1]: 30 sa_4 + 30 sa_5 >= 2.5
 con_beta_pw_TLMdkc[2,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[3,C1,P1,CPU1]: 30 sa_3 >= 1.25
 con_beta_pw_TLMdkc[3,C2,P1,CPU1]: 2 sa_10 + 2 sa_11 + 2 sa_12 + 2 sa_13
   + 10 sa_16 + 9 sa_17 + 10 sa_18 + 10 sa_19 + 9 sa_25 + 10 sa_26 >= 1.25
 con_beta_pw_Mdkc[4,C1,P1,CPU1]: 30 sa_6 >= 0.625
 con_beta_pw_Mdkc[4,C2,P1,CPU1]: 2 sa_2 + 10 sa_16 >= 0.625
 con_beta_ps_t0dkc[1,0,C1,P1,CPU1]: 14 sa_10 + 2 sa_16 + 30 sa_35
   + 30 sa_36 >= 5
 con_beta_ps_t0dkc[1,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C1,P1,CPU1]: 10 sa_16 + 10 sa_17 + 10 sa_18
   + 10 sa_19 + 10 sa_20 + 10 sa_21 + 10 sa_22 + 10 sa_23 + 10 sa_36
   >= 2.5
 con_beta_ps_t0dkc[2,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C1,P1,CPU1]: 30 sa_23 + 18 sa_25 + 6 sa_26 >= 1.25
 con_beta_ps_t0dkc[3,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[1,1,C1,P1,CPU1]: - 9.5 sa_16 - 9.5 sa_17 - 9.5 sa_18
   - 9.5 sa_19 - 9.5 sa_20 - 9.5 sa_21 - 9.5 sa_22 - 9.5 sa_23 + 30 sa_32
   + 30 sa_33 - 9.5 sa_36 >= 2.5
 con_beta_ps_t1TLdkc[1,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[2,1,C1,P1,CPU1]: 10 sa_1 >= 1.25
 con_beta_ps_t1TLdkc[2,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[3,1,C1,P1,CPU1]: 14 sa_13 + 30 sa_14 >= 0.625
 con_beta_ps_t1TLdkc[3,1,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[1,2,C1,P1,CPU1]: - 8.55 sa_1 + 14 sa_2 + 30 sa_30
   >= 1.25
 con_beta_ps_tTLMdkc[1,2,C2,P1,CPU1]: - 0.95 sa_1 + 10 sa_2 + 10 sa_1156
   + 10 sa_1157 >= 0
 con_beta_ps_tTLMdkc[1,3,C1,P1,CPU1]: - 8.55 sa_4 + 30 sa_31 + 2 sa_1156
   + 2 sa_1157 >= 0.625
 con_beta_ps_tTLMdkc[1,3,C2,P1,CPU1]: - 0.95 sa_4 + 10 sa_11 + 10 sa_12
   + 10 sa_13 + sa_1156 + sa_1157 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C1,P1,CPU1]: 10 sa_4 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[2,3,C1,P1,CPU1]: 10 sa_5 + 10 sa_33 >= 0.3125
 con_beta_ps_tTLMdkc[2,3,C2,P1,CPU1]: 6 sa_8 + 7 sa_34 + 7 sa_35 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C1,P1,CPU1]: 30 sa_9 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[3,3,C1,P1,CPU1]: 30 sa_8 >= 0.15625
 con_beta_ps_tTLMdkc[3,3,C2,P1,CPU1]: 10 sa_18 >= 0.15625
 con_beta_ps_tMdkc[1,4,C1,P1,CPU1]: - 8.55 sa_5 - 1.71 sa_8 - 8.55 sa_9
   - 8.55 sa_11 - 8.55 sa_33 + 30 sa_34 >= 0.3125
 con_beta_ps_tMdkc[1,4,C2,P1,CPU1]: - 0.95 sa_5 - 5.89 sa_8 - 0.95 sa_9
   + 10 sa_10 - 0.95 sa_11 - 6.65 sa_12 - 0.95 sa_33 - 6.65 sa_34
   - 6.65 sa_35 + 10 sa_1156 + 10 sa_1157 >= 0.3125
 con_beta_ps_tMdkc[2,4,C1,P1,CPU1]: 2 sa_8 + 10 sa_9 + 10 sa_11 >= 0.15625
 con_beta_ps_tMdkc[2,4,C2,P1,CPU1]: 7 sa_12 >= 0.15625
 con_beta_ps_tMdkc[3,4,C1,P1,CPU1]: 18 sa_17 >= 0.078125
 con_beta_ps_tMdkc[3,4,C2,P1,CPU1]: sa_18 + 10 sa_19 >= 0.078125
 con_beta_ps_Tmdkc[4,0,C1,P1,CPU1]: 30 sa_24 >= 0.625
 con_beta_ps_Tmdkc[4,0,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,1,C1,P1,CPU1]: 2 sa_19 + 18 sa_21 + 30 sa_22
   >= 0.3125
 con_beta_ps_Tmdkc[4,1,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,2,C1,P1,CPU1]: 30 sa_28 >= 0.15625
 con_beta_ps_Tmdkc[4,2,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,3,C1,P1,CPU1]: 2 sa_18 + 30 sa_27 >= 0.078125
 con_beta_ps_Tmdkc[4,3,C2,P1,CPU1]: sa_16 + sa_19 + 9 sa_21 >= 0.078125
 con_beta_ps_Tmdkc[4,4,C1,P1,CPU1]: 30 sa_20 >= 0.0390625
 con_beta_ps_Tmdkc[4,4,C2,P1,CPU1]: 8 sa_26 >= 0.0390625
Bounds
End
