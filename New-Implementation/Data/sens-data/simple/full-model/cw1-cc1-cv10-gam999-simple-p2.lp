\ Model Master problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 59.82011997000001 sa_38 - 29.91003 sa_44 + 29.97 sa_169 + 17.982 sa_174
   - 17.926055986 sa_175 - 59.82011997000001 sa_177 + 21.981998 sa_180
   + 9.008990999999998 sa_181 + 29.97 sa_182 + 29.97 sa_183 + 8.991 sa_184
   - 59.82011997000001 sa_189 - 59.82011997000001 sa_192
   - 53.83810797300001 sa_195 - 29.91003 sa_196 + 29.97 sa_198
   + 29.97 sa_199 + 1.02099 sa_202 + 18.984999 sa_203 + 12.002 sa_204
   + 0.018 sa_208 - 29.91003 sa_210 - 17.946018 sa_211 - 29.91003 sa_212
   - 29.91003 sa_213 + 0.03 sa_214 + 0.018 sa_220 - 19.93403999 sa_223
   - 59.82011997000001 sa_1424 - 59.82011997000001 sa_1425
Subject To
 con_beta_0: 0.001 sa_38 + 0.001 sa_44 + 0.001 sa_115 + 0.001 sa_169
   + 0.001 sa_174 + 0.001 sa_175 + 0.001 sa_176 + 0.001 sa_177
   + 0.001 sa_178 + 0.001 sa_180 + 0.001 sa_181 + 0.001 sa_182
   + 0.001 sa_183 + 0.001 sa_184 + 0.001 sa_189 + 0.001 sa_191
   + 0.001 sa_192 + 0.001 sa_193 + 0.001 sa_195 + 0.001 sa_196
   + 0.001 sa_198 + 0.001 sa_199 + 0.001 sa_202 + 0.001 sa_203
   + 0.001 sa_204 + 0.001 sa_208 + 0.001 sa_210 + 0.001 sa_211
   + 0.001 sa_212 + 0.001 sa_213 + 0.001 sa_214 + 0.001 sa_220
   + 0.001 sa_223 + 0.001 sa_1424 + 0.001 sa_1425 = 1
 con_beta_ul_nco[PPE1]: 60 sa_38 + 60 sa_44 + 60 sa_115 + 60 sa_169
   + 60 sa_174 + 60 sa_175 + 60 sa_176 + 60 sa_177 + 60 sa_178 + 60 sa_180
   + 60 sa_181 + 60 sa_182 + 60 sa_183 + 60 sa_184 + 60 sa_189 + 60 sa_191
   + 60 sa_192 + 60 sa_193 + 60 sa_195 + 60 sa_196 + 60 sa_198 + 60 sa_199
   + 60 sa_202 + 60 sa_203 + 60 sa_204 + 60 sa_208 + 60 sa_210 + 60 sa_211
   + 60 sa_212 + 60 sa_213 + 60 sa_214 + 60 sa_220 + 60 sa_223 + 60 sa_1424
   + 60 sa_1425 >= 0
 con_beta_pw_0dkc[0,C1,P1,CPU1]: - 8.991 sa_38 - 8.991 sa_44 - 8.991 sa_115
   + 21.009 sa_169 - 8.991 sa_174 - 8.991 sa_175 - 8.991 sa_176
   - 8.991 sa_177 - 8.991 sa_178 - 8.991 sa_180 - 8.991 sa_181
   - 8.991 sa_182 - 8.991 sa_183 - 8.991 sa_184 - 8.991 sa_189
   - 8.991 sa_191 - 8.991 sa_192 - 8.991 sa_193 - 8.991 sa_195
   - 8.991 sa_196 - 8.991 sa_198 - 8.991 sa_199 - 8.991 sa_202
   - 8.991 sa_203 - 8.991 sa_204 - 8.991 sa_208 - 8.991 sa_210
   - 8.991 sa_211 - 8.991 sa_212 - 8.991 sa_213 - 8.991 sa_214
   - 8.991 sa_220 - 8.991 sa_223 - 8.991 sa_1424 - 8.991 sa_1425 >= 10
 con_beta_pw_0dkc[0,C2,P1,CPU1]: >= 0
 con_beta_pw_1TLdkc[1,C1,P1,CPU1]: 30 sa_183 >= 5
 con_beta_pw_1TLdkc[1,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[2,C1,P1,CPU1]: 14 sa_180 + 18 sa_181 + 30 sa_182
   >= 2.5
 con_beta_pw_TLMdkc[2,C2,P1,CPU1]: >= 0
 con_beta_pw_TLMdkc[3,C1,P1,CPU1]: 18 sa_174 + 30 sa_199 >= 1.25
 con_beta_pw_TLMdkc[3,C2,P1,CPU1]: 10 sa_175 + 9 sa_184 + sa_202
   + 10 sa_203 + 2 sa_204 >= 1.25
 con_beta_pw_Mdkc[4,C1,P1,CPU1]: 30 sa_198 >= 0.625
 con_beta_pw_Mdkc[4,C2,P1,CPU1]: 10 sa_180 + 10 sa_202 + 10 sa_203
   + 10 sa_204 >= 0.625
 con_beta_ps_t0dkc[1,0,C1,P1,CPU1]: 30 sa_178 >= 5
 con_beta_ps_t0dkc[1,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[2,0,C1,P1,CPU1]: 10 sa_177 + 10 sa_178 >= 2.5
 con_beta_ps_t0dkc[2,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t0dkc[3,0,C1,P1,CPU1]: 30 sa_213 >= 1.25
 con_beta_ps_t0dkc[3,0,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[1,1,C1,P1,CPU1]: 30 sa_176 - 9.99 sa_177 - 9.99 sa_178
   + 30 sa_193 >= 2.5
 con_beta_ps_t1TLdkc[1,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[2,1,C1,P1,CPU1]: 30 sa_214 + 10 sa_1424 + 10 sa_1425
   >= 1.25
 con_beta_ps_t1TLdkc[2,1,C2,P1,CPU1]: >= 0
 con_beta_ps_t1TLdkc[3,1,C1,P1,CPU1]: 18 sa_211 + 30 sa_212 >= 0.625
 con_beta_ps_t1TLdkc[3,1,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[1,2,C1,P1,CPU1]: 18 sa_184 - 8.991 sa_1424
   - 8.991 sa_1425 >= 1.25
 con_beta_ps_tTLMdkc[1,2,C2,P1,CPU1]: - 0.999 sa_1424 - 0.999 sa_1425 >= 0
 con_beta_ps_tTLMdkc[1,3,C1,P1,CPU1]: 18 sa_115 >= 0.625
 con_beta_ps_tTLMdkc[1,3,C2,P1,CPU1]: 9 sa_174 + 2 sa_175 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C1,P1,CPU1]: 18 sa_220 + 6 sa_223 >= 0.625
 con_beta_ps_tTLMdkc[2,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[2,3,C1,P1,CPU1]: 2 sa_202 + 2 sa_203 + 14 sa_204
   >= 0.3125
 con_beta_ps_tTLMdkc[2,3,C2,P1,CPU1]: 7 sa_175 + 7 sa_189 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C1,P1,CPU1]: 30 sa_196 >= 0.3125
 con_beta_ps_tTLMdkc[3,2,C2,P1,CPU1]: >= 0
 con_beta_ps_tTLMdkc[3,3,C1,P1,CPU1]: 30 sa_210 >= 0.15625
 con_beta_ps_tTLMdkc[3,3,C2,P1,CPU1]: 10 sa_202 + sa_203 >= 0.15625
 con_beta_ps_tMdkc[1,4,C1,P1,CPU1]: 30 sa_191 >= 0.3125
 con_beta_ps_tMdkc[1,4,C2,P1,CPU1]: 9 sa_115 - 6.993 sa_175 - 6.993 sa_176
   - 6.993 sa_189 + 9 sa_208 + 9 sa_211 + 9 sa_220 + 8 sa_223 >= 0.3125
 con_beta_ps_tMdkc[2,4,C1,P1,CPU1]: 18 sa_208 >= 0.15625
 con_beta_ps_tMdkc[2,4,C2,P1,CPU1]: 7 sa_176 >= 0.15625
 con_beta_ps_tMdkc[3,4,C1,P1,CPU1]: 30 sa_44 >= 0.078125
 con_beta_ps_tMdkc[3,4,C2,P1,CPU1]: 2 sa_180 + 9 sa_181 >= 0.078125
 con_beta_ps_Tmdkc[4,0,C1,P1,CPU1]: 14 sa_175 + 18 sa_195 >= 0.625
 con_beta_ps_Tmdkc[4,0,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,1,C1,P1,CPU1]: 30 sa_38 >= 0.3125
 con_beta_ps_Tmdkc[4,1,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,2,C1,P1,CPU1]: 30 sa_189 >= 0.15625
 con_beta_ps_Tmdkc[4,2,C2,P1,CPU1]: >= 0
 con_beta_ps_Tmdkc[4,3,C1,P1,CPU1]: 30 sa_177 >= 0.078125
 con_beta_ps_Tmdkc[4,3,C2,P1,CPU1]: 9 sa_195 >= 0.078125
 con_beta_ps_Tmdkc[4,4,C1,P1,CPU1]: 30 sa_192 + 30 sa_1424 + 30 sa_1425
   >= 0.0390625
 con_beta_ps_Tmdkc[4,4,C2,P1,CPU1]: 10 sa_223 >= 0.0390625
Bounds
End
