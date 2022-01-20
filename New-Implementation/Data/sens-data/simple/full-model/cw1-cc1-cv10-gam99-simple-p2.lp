\ Model Master problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 58.21197000000001 sa_38 - 58.21197000000001 sa_42 - 43.658082 sa_43
   - 58.21197000000001 sa_44 - 58.21197000000001 sa_48 - 29.103 sa_49
   + 0.99 sa_55 + 17.82 sa_113 + 29.7 sa_114 + 29.7 sa_127 + 29.7 sa_133
   + 29.7 sa_134 + 0.356409 sa_135 + 29.7 sa_136 + 9.09 sa_137
   - 29.103 sa_139 + 8.91 sa_142 - 8.5518 sa_143 + 8.91 sa_144
   - 29.103 sa_148 - 29.103 sa_153 + 0.3 sa_155 + 0.3 sa_157 + 0.3 sa_159
   - 43.658082 sa_162 - 21.34419 sa_271 - 58.21197000000001 sa_1386
   - 58.21197000000001 sa_1387
Subject To
 con_beta_0: 0.01 sa_38 + 0.01 sa_42 + 0.01 sa_43 + 0.01 sa_44 + 0.01 sa_48
   + 0.01 sa_49 + 0.01 sa_55 + 0.01 sa_102 + 0.01 sa_103 + 0.01 sa_113
   + 0.01 sa_114 + 0.01 sa_127 + 0.01 sa_133 + 0.01 sa_134 + 0.01 sa_135
   + 0.01 sa_136 + 0.01 sa_137 + 0.01 sa_138 + 0.01 sa_139 + 0.01 sa_142
   + 0.01 sa_143 + 0.01 sa_144 + 0.01 sa_145 + 0.01 sa_146 + 0.01 sa_148
   + 0.01 sa_153 + 0.01 sa_155 + 0.01 sa_157 + 0.01 sa_159 + 0.01 sa_162
   + 0.01 sa_163 + 0.01 sa_271 + 0.01 sa_1386 + 0.01 sa_1387 = 1
 con_beta_ul_nco[PPE1]: 60 sa_38 + 60 sa_42 + 60 sa_43 + 60 sa_44
   + 60 sa_48 + 60 sa_49 + 60 sa_55 + 60 sa_102 + 60 sa_103 + 60 sa_113
   + 60 sa_114 + 60 sa_127 + 60 sa_133 + 60 sa_134 + 60 sa_135 + 60 sa_136
   + 60 sa_137 + 60 sa_138 + 60 sa_139 + 60 sa_142 + 60 sa_143 + 60 sa_144
   + 60 sa_145 + 60 sa_146 + 60 sa_148 + 60 sa_153 + 60 sa_155 + 60 sa_157
   + 60 sa_159 + 60 sa_162 + 60 sa_163 + 60 sa_271 + 60 sa_1386
   + 60 sa_1387 >= 0
 con_beta_pw_0dkc[0,C1,P1,CPU1]: - 8.91 sa_38 - 8.91 sa_42 - 8.91 sa_43
   - 8.91 sa_44 - 8.91 sa_48 - 8.91 sa_49 - 8.91 sa_55 - 8.91 sa_102
   - 8.91 sa_103 + 9.09 sa_113 + 21.09 sa_114 - 8.91 sa_127 - 8.91 sa_133
   - 8.91 sa_134 - 8.91 sa_135 - 8.91 sa_136 - 8.91 sa_137 - 8.91 sa_138
   - 8.91 sa_139 - 8.91 sa_142 - 8.91 sa_143 - 8.91 sa_144 - 8.91 sa_145
   - 8.91 sa_146 - 8.91 sa_148 - 8.91 sa_153 - 8.91 sa_155 - 8.91 sa_157
   - 8.91 sa_159 - 8.91 sa_162 - 8.91 sa_163 - 8.91 sa_271 - 8.91 sa_1386
   - 8.91 sa_1387 >= 10
 con_beta_pw_0dkc[0,C2,P1,CPU1]: >= 0
 con_beta_pw_1TLdkc[1,C1,P1,CPU1]: 30 sa_133 >= 5
 con_beta_pw_1TLdkc[1,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[2,C1,P1,CPU1]: 18 sa_135 + 30 sa_136 >= 2.5
 con_beta_pw_TLMdkc[2,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[3,C1,P1,CPU1]: 30 sa_127 >= 1.25
 con_beta_pw_TLMdkc[3,C2,P1,CPU1]: 9 sa_143 + 9 sa_144 >= 1.25
 con_beta_pw_Mdkc[4,C1,P1,CPU1]: 30 sa_134 >= 0.625
 con_beta_pw_Mdkc[4,C2,P1,CPU1]: sa_55 + 9 sa_137 + 9 sa_142 >= 0.625
 con_beta_ps_t0dkc[1,0,C1,P1,CPU1]: 18 sa_142 + 30 sa_145 + 30 sa_146 >= 5
 con_beta_ps_t0dkc[1,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C1,P1,CPU1]: 2 sa_42 + 2 sa_43 + 10 sa_44 >= 2.5
 con_beta_ps_t0dkc[2,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C1,P1,CPU1]: 30 sa_49 >= 1.25
 con_beta_ps_t0dkc[3,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[1,1,C1,P1,CPU1]: - 1.98 sa_42 - 1.98 sa_43 - 9.9 sa_44
   + 18 sa_144 + 30 sa_163 + 14 sa_271 >= 2.5
 con_beta_ps_t1TLdkc[1,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[2,1,C1,P1,CPU1]: 30 sa_157 + 10 sa_1386 + 10 sa_1387
   >= 1.25
 con_beta_ps_t1TLdkc[2,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[3,1,C1,P1,CPU1]: 30 sa_148 >= 0.625
 con_beta_ps_t1TLdkc[3,1,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[1,2,C1,P1,CPU1]: 2 sa_55 + 30 sa_102 - 8.91 sa_1386
   - 8.91 sa_1387 >= 1.25
 con_beta_ps_tTLMdkc[1,2,C2,P1,CPU1]: - 0.99 sa_1386 - 0.99 sa_1387 >= 0
 con_beta_ps_tTLMdkc[1,3,C1,P1,CPU1]: 30 sa_103 >= 0.625
 con_beta_ps_tTLMdkc[1,3,C2,P1,CPU1]: 10 sa_55 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C1,P1,CPU1]: 30 sa_159 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[2,3,C1,P1,CPU1]: 18 sa_137 >= 0.3125
 con_beta_ps_tTLMdkc[2,3,C2,P1,CPU1]: 6 sa_42 + 6 sa_43 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C1,P1,CPU1]: 30 sa_153 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[3,3,C1,P1,CPU1]: 18 sa_143 >= 0.15625
 con_beta_ps_tTLMdkc[3,3,C2,P1,CPU1]: 9 sa_162 + 2 sa_271 >= 0.15625
 con_beta_ps_tMdkc[1,4,C1,P1,CPU1]: 30 sa_138 >= 0.3125
 con_beta_ps_tMdkc[1,4,C2,P1,CPU1]: - 5.94 sa_42 - 5.94 sa_43 + 10 sa_55
   + 9 sa_113 - 6.93 sa_145 >= 0.3125
 con_beta_ps_tMdkc[2,4,C1,P1,CPU1]: 30 sa_155 >= 0.15625
 con_beta_ps_tMdkc[2,4,C2,P1,CPU1]: 7 sa_145 >= 0.15625
 con_beta_ps_tMdkc[3,4,C1,P1,CPU1]: 30 sa_139 >= 0.078125
 con_beta_ps_tMdkc[3,4,C2,P1,CPU1]: 9 sa_43 >= 0.078125
 con_beta_ps_Tmdkc[4,0,C1,P1,CPU1]: 18 sa_43 + 30 sa_44 >= 0.625
 con_beta_ps_Tmdkc[4,0,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,1,C1,P1,CPU1]: 30 sa_38 >= 0.3125
 con_beta_ps_Tmdkc[4,1,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,2,C1,P1,CPU1]: 30 sa_42 >= 0.15625
 con_beta_ps_Tmdkc[4,2,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,3,C1,P1,CPU1]: 18 sa_162 >= 0.078125
 con_beta_ps_Tmdkc[4,3,C2,P1,CPU1]: 9 sa_135 >= 0.078125
 con_beta_ps_Tmdkc[4,4,C1,P1,CPU1]: 30 sa_48 + 30 sa_1386 + 30 sa_1387
   >= 0.0390625
 con_beta_ps_Tmdkc[4,4,C2,P1,CPU1]: 10 sa_271 >= 0.0390625
Bounds
End
