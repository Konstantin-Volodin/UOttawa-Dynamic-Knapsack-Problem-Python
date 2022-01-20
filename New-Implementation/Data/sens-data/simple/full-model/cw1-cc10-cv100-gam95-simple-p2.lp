\ Model Master problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  28.5 sa_0 + 17.1 sa_1 + 15.2 sa_2 + 28.5 sa_3 + 28.5 sa_4 + 28.5 sa_5
   + 28.5 sa_6 + 28.5 sa_7 + 28.5 sa_8 + 28.5 sa_9 + 2.85 sa_10
   + 11.4 sa_11 + 28.5 sa_12
Subject To
 con_beta_0: 0.05 sa_0 + 0.05 sa_1 + 0.05 sa_2 + 0.05 sa_3 + 0.05 sa_4
   + 0.05 sa_5 + 0.05 sa_6 + 0.05 sa_7 + 0.05 sa_8 + 0.05 sa_9 + 0.05 sa_10
   + 0.05 sa_11 + 0.05 sa_12 + 0.05 sa_13 + 0.05 sa_14 + 0.05 sa_15
   + 0.05 sa_16 + 0.05 sa_17 + 0.05 sa_18 + 0.05 sa_19 + 0.05 sa_20
   + 0.05 sa_22 + 0.05 sa_23 + 0.05 sa_24 + 0.05 sa_25 + 0.05 sa_26
   + 0.05 sa_27 + 0.05 sa_28 + 0.05 sa_29 + 0.05 sa_30 + 0.05 sa_31
   + 0.05 sa_32 + 0.05 sa_33 + 0.05 sa_34 + 0.05 sa_35 + 0.05 sa_36
   + 0.05 sa_37 + 0.05 sa_38 + 0.05 sa_39 + 0.05 sa_40 + 0.05 sa_41
   + 0.05 sa_42 + 0.05 sa_43 + 0.05 sa_44 + 0.05 sa_45 + 0.05 sa_46
   + 0.05 sa_47 + 0.05 sa_48 + 0.05 sa_49 + 0.05 sa_50 + 0.05 sa_51
   + 0.05 sa_52 + 0.05 sa_53 + 0.05 sa_54 + 0.05 sa_55 + 0.05 sa_56
   + 0.05 sa_57 + 0.05 sa_58 + 0.05 sa_59 + 0.05 sa_60 + 0.05 sa_61
   + 0.05 sa_62 + 0.05 sa_63 + 0.05 sa_64 + 0.05 sa_65 + 0.05 sa_66
   + 0.05 sa_67 + 0.05 sa_68 + 0.05 sa_69 + 0.05 sa_70 + 0.05 sa_71
   + 0.05 sa_72 + 0.05 sa_73 + 0.05 sa_74 + 0.05 sa_75 + 0.05 sa_76
   + 0.05 sa_77 + 0.05 sa_78 + 0.05 sa_79 + 0.05 sa_80 + 0.05 sa_81
   + 0.05 sa_82 + 0.05 sa_83 + 0.05 sa_84 + 0.05 sa_85 + 0.05 sa_86
   + 0.05 sa_87 + 0.05 sa_88 + 0.05 sa_89 + 0.05 sa_90 + 0.05 sa_91
   + 0.05 sa_92 + 0.05 sa_93 + 0.05 sa_94 + 0.05 sa_95 + 0.05 sa_96
   + 0.05 sa_97 + 0.05 sa_98 + 0.05 sa_99 + 0.05 sa_100 + 0.05 sa_101
   + 0.05 sa_102 + 0.05 sa_103 + 0.05 sa_104 + 0.05 sa_105 + 0.05 sa_106
   + 0.05 sa_107 + 0.05 sa_108 + 0.05 sa_109 + 0.05 sa_110 + 0.05 sa_111
   + 0.05 sa_112 + 0.05 sa_113 + 0.05 sa_114 + 0.05 sa_115 + 0.05 sa_116
   + 0.05 sa_117 + 0.05 sa_118 + 0.05 sa_119 + 0.05 sa_120 + 0.05 sa_121
   + 0.05 sa_122 + 0.05 sa_123 = 1
 con_beta_ul_nco[PPE1]: 60 sa_0 + 60 sa_1 + 60 sa_2 + 60 sa_3 + 60 sa_4
   + 60 sa_5 + 60 sa_6 + 60 sa_7 + 60 sa_8 + 60 sa_9 + 60 sa_10 + 60 sa_11
   + 60 sa_12 + 60 sa_14 + 60 sa_15 + 60 sa_16 + 60 sa_17 + 60 sa_18
   + 60 sa_19 + 60 sa_20 >= 0
 con_beta_pw_0dkc[0,C1,P1,CPU1]: - 8.55 sa_0 + 9.45 sa_1 + 5.45 sa_2
   + 21.45 sa_3 + 21.45 sa_4 + 21.45 sa_5 + 21.45 sa_6 + 21.45 sa_7
   - 8.55 sa_8 - 8.55 sa_9 - 8.55 sa_10 - 8.55 sa_11 - 8.55 sa_12
   - 8.55 sa_13 - 8.55 sa_14 - 8.55 sa_15 - 8.55 sa_16 - 8.55 sa_17
   - 8.55 sa_18 - 8.55 sa_19 - 8.55 sa_20 - 8.55 sa_22 - 8.55 sa_23
   - 8.55 sa_24 - 8.55 sa_25 - 8.55 sa_26 - 8.55 sa_27 - 8.55 sa_28
   - 8.55 sa_29 - 8.55 sa_30 - 8.55 sa_31 - 8.55 sa_32 - 8.55 sa_33
   - 8.55 sa_34 - 8.55 sa_35 - 8.55 sa_36 - 8.55 sa_37 - 8.55 sa_38
   - 8.55 sa_39 - 8.55 sa_40 - 8.55 sa_41 - 8.55 sa_42 - 8.55 sa_43
   - 8.55 sa_44 - 8.55 sa_45 - 8.55 sa_46 - 8.55 sa_47 - 8.55 sa_48
   - 8.55 sa_49 - 8.55 sa_50 - 8.55 sa_51 - 8.55 sa_52 - 8.55 sa_53
   - 8.55 sa_54 - 8.55 sa_55 - 8.55 sa_56 - 8.55 sa_57 - 8.55 sa_58
   - 8.55 sa_59 - 8.55 sa_60 - 8.55 sa_61 - 8.55 sa_62 - 8.55 sa_63
   - 8.55 sa_64 - 8.55 sa_65 - 8.55 sa_66 - 8.55 sa_67 - 8.55 sa_68
   - 8.55 sa_69 - 8.55 sa_70 - 8.55 sa_71 - 8.55 sa_72 - 8.55 sa_73
   - 8.55 sa_74 - 8.55 sa_75 - 8.55 sa_76 - 8.55 sa_77 - 8.55 sa_78
   - 8.55 sa_79 - 8.55 sa_80 - 8.55 sa_81 - 8.55 sa_82 - 8.55 sa_83
   - 8.55 sa_84 - 8.55 sa_85 - 8.55 sa_86 - 8.55 sa_87 - 8.55 sa_88
   - 8.55 sa_89 - 8.55 sa_90 - 8.55 sa_91 - 8.55 sa_92 - 8.55 sa_93
   - 8.55 sa_94 - 8.55 sa_95 - 8.55 sa_96 - 8.55 sa_97 - 8.55 sa_98
   - 8.55 sa_99 - 8.55 sa_100 - 8.55 sa_101 - 8.55 sa_102 - 8.55 sa_103
   - 8.55 sa_104 - 8.55 sa_105 - 8.55 sa_106 - 8.55 sa_107 - 8.55 sa_108
   - 8.55 sa_109 - 8.55 sa_110 - 8.55 sa_111 - 8.55 sa_112 - 8.55 sa_113
   - 8.55 sa_114 - 8.55 sa_115 - 8.55 sa_116 - 8.55 sa_117 - 8.55 sa_118
   - 8.55 sa_119 - 8.55 sa_120 - 8.55 sa_121 - 8.55 sa_122 - 8.55 sa_123
   >= 10
 con_beta_pw_0dkc[0,C2,P1,CPU1]: >= 0
 con_beta_pw_1TLdkc[1,C1,P1,CPU1]: 30 sa_9 + 3 sa_10 >= 5
 con_beta_pw_1TLdkc[1,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[2,C1,P1,CPU1]: 2 sa_11 + 30 sa_12 >= 2.5
 con_beta_pw_TLMdkc[2,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[3,C1,P1,CPU1]: 30 sa_8 >= 1.25
 con_beta_pw_TLMdkc[3,C2,P1,CPU1]: 10 sa_11 >= 1.25
 con_beta_pw_Mdkc[4,C1,P1,CPU1]: 30 sa_0 >= 0.625
 con_beta_pw_Mdkc[4,C2,P1,CPU1]: 2 sa_2 >= 0.625
 con_beta_ps_t0dkc[1,0,C1,P1,CPU1]: 30 sa_19 >= 5
 con_beta_ps_t0dkc[1,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C1,P1,CPU1]: 10 sa_6 >= 2.5
 con_beta_ps_t0dkc[2,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C1,P1,CPU1]: 10 sa_1 >= 1.25
 con_beta_ps_t0dkc[3,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[1,1,C1,P1,CPU1]: - 9.5 sa_6 + 30 sa_18 >= 2.5
 con_beta_ps_t1TLdkc[1,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[2,1,C1,P1,CPU1]: 10 sa_0 - 9.5 sa_1 + 2 sa_3 + 10 sa_8
   + 10 sa_9 + 10 sa_10 + 10 sa_11 + 10 sa_17 + 10 sa_18 + 10 sa_19
   >= 1.25
 con_beta_ps_t1TLdkc[2,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[3,1,C1,P1,CPU1]: 10 sa_2 - 9.5 sa_11 - 9.5 sa_12
   - 9.5 sa_20 >= 0.625
 con_beta_ps_t1TLdkc[3,1,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[1,2,C1,P1,CPU1]: - 8.55 sa_0 - 1.71 sa_3 - 8.55 sa_8
   - 8.55 sa_9 - 8.55 sa_10 - 8.55 sa_11 + 21.45 sa_17 - 8.55 sa_18
   - 8.55 sa_19 + 30 sa_20 >= 1.25
 con_beta_ps_tTLMdkc[1,2,C2,P1,CPU1]: - 0.95 sa_0 - 0.19 sa_3 - 0.95 sa_8
   - 0.95 sa_9 - 0.95 sa_10 - 0.95 sa_11 + 9 sa_15 + 9 sa_16 - 0.95 sa_17
   - 0.95 sa_18 - 0.95 sa_19 >= 0
 con_beta_ps_tTLMdkc[1,3,C1,P1,CPU1]: - 8.55 sa_1 - 1.71 sa_14 + 18 sa_15
   + 18 sa_16 >= 0.625
 con_beta_ps_tTLMdkc[1,3,C2,P1,CPU1]: - 0.95 sa_1 - 5.7 sa_5 + 10 sa_10
   + 10 sa_11 - 0.19 sa_14 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C1,P1,CPU1]: 10 sa_1 - 8.55 sa_2 + 2 sa_14
   >= 0.625
 con_beta_ps_tTLMdkc[2,2,C2,P1,CPU1]: - 0.95 sa_2 + 6 sa_5 >= 0
 con_beta_ps_tTLMdkc[2,3,C1,P1,CPU1]: - 8.55 sa_3 - 8.55 sa_4 + 10 sa_7
   + 10 sa_15 + 10 sa_16 >= 0.3125
 con_beta_ps_tTLMdkc[2,3,C2,P1,CPU1]: 7 sa_2 + 5.05 sa_3 - 0.95 sa_4
   + 7 sa_12 - 5.7 sa_17 + 7 sa_20 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C1,P1,CPU1]: 10 sa_3 + 1.45 sa_4 - 8.55 sa_5
   >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C2,P1,CPU1]: - 0.95 sa_4 - 0.95 sa_5 + 6 sa_17
   >= 0
 con_beta_ps_tTLMdkc[3,3,C1,P1,CPU1]: 10 sa_0 - 8.55 sa_2 - 8.55 sa_3
   + 10 sa_6 + 10 sa_12 + 2 sa_16 + 2 sa_17 + 10 sa_18 + 10 sa_20
   >= 0.15625
 con_beta_ps_tTLMdkc[3,3,C2,P1,CPU1]: - 0.95 sa_2 - 0.95 sa_3 + 7 sa_15
   + 6 sa_16 >= 0.15625
 con_beta_ps_tMdkc[1,4,C1,P1,CPU1]: - 8.55 sa_4 - 1.71 sa_5 - 8.55 sa_7
   + 30 sa_14 - 8.55 sa_15 - 8.55 sa_16 >= 0.3125
 con_beta_ps_tMdkc[1,4,C2,P1,CPU1]: 9 sa_1 + 3.35 sa_2 - 5.7 sa_3
   - 0.95 sa_4 - 0.19 sa_5 - 0.95 sa_7 + 10 sa_10 + sa_11 - 6.65 sa_12
   - 5.7 sa_14 - 0.95 sa_15 - 0.95 sa_16 - 6.65 sa_20 >= 0.3125
 con_beta_ps_tMdkc[2,4,C1,P1,CPU1]: - 8.55 sa_0 + 10 sa_4 - 6.55 sa_5
   - 8.55 sa_6 - 8.55 sa_7 - 8.55 sa_12 - 1.71 sa_16 - 1.71 sa_17
   - 8.55 sa_18 - 8.55 sa_20 >= 0.15625
 con_beta_ps_tMdkc[2,4,C2,P1,CPU1]: - 0.95 sa_0 - 0.95 sa_5 - 0.95 sa_6
   - 0.95 sa_7 - 6.65 sa_8 - 6.65 sa_9 - 6.65 sa_10 - 6.65 sa_11
   - 0.95 sa_12 + 6 sa_14 - 6.65 sa_15 - 5.89 sa_16 - 0.19 sa_17
   - 0.95 sa_18 - 6.65 sa_19 - 0.95 sa_20 >= 0.15625
 con_beta_ps_tMdkc[3,4,C1,P1,CPU1]: - 8.55 sa_1 + 10 sa_5 - 8.55 sa_6
   + 10 sa_7 >= 0.078125
 con_beta_ps_tMdkc[3,4,C2,P1,CPU1]: - 6.65 sa_0 - 0.95 sa_1 - 0.95 sa_6
   + 7 sa_8 + 7 sa_9 + 7 sa_10 + 7 sa_11 - 6.65 sa_16 - 6.65 sa_17
   + 7 sa_19 >= 0.078125
 con_beta_ps_Tmdkc[4,0,C1,P1,CPU1]: 10 sa_11 + 10 sa_12 + 10 sa_20
   >= 0.625
 con_beta_ps_Tmdkc[4,0,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,1,C1,P1,CPU1]: 10 sa_4 + 10 sa_5 >= 0.3125
 con_beta_ps_Tmdkc[4,1,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,2,C1,P1,CPU1]: 10 sa_2 + 10 sa_3 >= 0.15625
 con_beta_ps_Tmdkc[4,2,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,3,C1,P1,CPU1]: 10 sa_6 >= 0.078125
 con_beta_ps_Tmdkc[4,3,C2,P1,CPU1]: 7 sa_16 + 7 sa_17 >= 0.078125
 con_beta_ps_Tmdkc[4,4,C1,P1,CPU1]: 10 sa_1 >= 0.0390625
 con_beta_ps_Tmdkc[4,4,C2,P1,CPU1]: 7 sa_0 >= 0.0390625
Bounds
End
