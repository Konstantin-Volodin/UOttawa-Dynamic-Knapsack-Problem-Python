\ Model Sub problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 con_auxiliary_uup[1,PPE1]: var_auxiliary_uup[1,PPE1]
   - 3 var_auxiliary_psp[1,0,C1,P1,CPU1]
   - 4 var_auxiliary_psp[1,0,C2,P1,CPU1]
   - 3 var_auxiliary_psp[1,1,C1,P1,CPU1]
   - 4 var_auxiliary_psp[1,1,C2,P1,CPU1]
   - 3 var_auxiliary_psp[1,2,C1,P1,CPU1]
   - 4 var_auxiliary_psp[1,2,C2,P1,CPU1]
   - 3 var_auxiliary_psp[1,3,C1,P1,CPU1]
   - 4 var_auxiliary_psp[1,3,C2,P1,CPU1]
   - 3 var_auxiliary_psp[1,4,C1,P1,CPU1]
   - 4 var_auxiliary_psp[1,4,C2,P1,CPU1] = 0
 con_auxiliary_uup[2,PPE1]: var_auxiliary_uup[2,PPE1]
   - 3 var_auxiliary_psp[2,0,C1,P1,CPU1]
   - 4 var_auxiliary_psp[2,0,C2,P1,CPU1]
   - 3 var_auxiliary_psp[2,1,C1,P1,CPU1]
   - 4 var_auxiliary_psp[2,1,C2,P1,CPU1]
   - 3 var_auxiliary_psp[2,2,C1,P1,CPU1]
   - 4 var_auxiliary_psp[2,2,C2,P1,CPU1]
   - 3 var_auxiliary_psp[2,3,C1,P1,CPU1]
   - 4 var_auxiliary_psp[2,3,C2,P1,CPU1]
   - 3 var_auxiliary_psp[2,4,C1,P1,CPU1]
   - 4 var_auxiliary_psp[2,4,C2,P1,CPU1] = 0
 con_auxiliary_uup[3,PPE1]: var_auxiliary_uup[3,PPE1]
   - 3 var_auxiliary_psp[3,0,C1,P1,CPU1]
   - 4 var_auxiliary_psp[3,0,C2,P1,CPU1]
   - 3 var_auxiliary_psp[3,1,C1,P1,CPU1]
   - 4 var_auxiliary_psp[3,1,C2,P1,CPU1]
   - 3 var_auxiliary_psp[3,2,C1,P1,CPU1]
   - 4 var_auxiliary_psp[3,2,C2,P1,CPU1]
   - 3 var_auxiliary_psp[3,3,C1,P1,CPU1]
   - 4 var_auxiliary_psp[3,3,C2,P1,CPU1]
   - 3 var_auxiliary_psp[3,4,C1,P1,CPU1]
   - 4 var_auxiliary_psp[3,4,C2,P1,CPU1] = 0
 con_auxiliary_uup[4,PPE1]: var_auxiliary_uup[4,PPE1]
   - 3 var_auxiliary_psp[4,0,C1,P1,CPU1]
   - 4 var_auxiliary_psp[4,0,C2,P1,CPU1]
   - 3 var_auxiliary_psp[4,1,C1,P1,CPU1]
   - 4 var_auxiliary_psp[4,1,C2,P1,CPU1]
   - 3 var_auxiliary_psp[4,2,C1,P1,CPU1]
   - 4 var_auxiliary_psp[4,2,C2,P1,CPU1]
   - 3 var_auxiliary_psp[4,3,C1,P1,CPU1]
   - 4 var_auxiliary_psp[4,3,C2,P1,CPU1]
   - 3 var_auxiliary_psp[4,4,C1,P1,CPU1]
   - 4 var_auxiliary_psp[4,4,C2,P1,CPU1] = 0
 con_auxiliary_pwp[0,C1,P1,CPU1]: - var_state_pw[0,C1,P1,CPU1]
   + var_action_sc[1,0,C1,P1,CPU1] + var_action_sc[2,0,C1,P1,CPU1]
   + var_action_sc[3,0,C1,P1,CPU1] + var_action_sc[4,0,C1,P1,CPU1]
   + var_auxiliary_pwp[0,C1,P1,CPU1] = 0
 con_auxiliary_pwp[0,C2,P1,CPU1]: - var_state_pw[0,C2,P1,CPU1]
   + var_action_sc[1,0,C2,P1,CPU1] + var_action_sc[2,0,C2,P1,CPU1]
   + var_action_sc[3,0,C2,P1,CPU1] + var_action_sc[4,0,C2,P1,CPU1]
   + var_auxiliary_pwp[0,C2,P1,CPU1] = 0
 con_auxiliary_pwp[1,C1,P1,CPU1]: - var_state_pw[1,C1,P1,CPU1]
   + var_action_sc[1,1,C1,P1,CPU1] + var_action_sc[2,1,C1,P1,CPU1]
   + var_action_sc[3,1,C1,P1,CPU1] + var_action_sc[4,1,C1,P1,CPU1]
   + var_auxiliary_pwp[1,C1,P1,CPU1] = 0
 con_auxiliary_pwp[1,C2,P1,CPU1]: - var_state_pw[1,C2,P1,CPU1]
   + var_action_sc[1,1,C2,P1,CPU1] + var_action_sc[2,1,C2,P1,CPU1]
   + var_action_sc[3,1,C2,P1,CPU1] + var_action_sc[4,1,C2,P1,CPU1]
   + var_auxiliary_pwp[1,C2,P1,CPU1] = 0
 con_auxiliary_pwp[2,C1,P1,CPU1]: - var_state_pw[2,C1,P1,CPU1]
   + var_action_sc[1,2,C1,P1,CPU1] + var_action_sc[2,2,C1,P1,CPU1]
   + var_action_sc[3,2,C1,P1,CPU1] + var_action_sc[4,2,C1,P1,CPU1]
   + var_auxiliary_pwp[2,C1,P1,CPU1] = 0
 con_auxiliary_pwp[2,C2,P1,CPU1]: - var_state_pw[2,C2,P1,CPU1]
   + var_action_sc[1,2,C2,P1,CPU1] + var_action_sc[2,2,C2,P1,CPU1]
   + var_action_sc[3,2,C2,P1,CPU1] + var_action_sc[4,2,C2,P1,CPU1]
   + var_auxiliary_pwp[2,C2,P1,CPU1] = 0
 con_auxiliary_pwp[3,C1,P1,CPU1]: - var_state_pw[3,C1,P1,CPU1]
   + var_action_sc[1,3,C1,P1,CPU1] + var_action_sc[2,3,C1,P1,CPU1]
   + var_action_sc[3,3,C1,P1,CPU1] + var_action_sc[4,3,C1,P1,CPU1]
   + var_auxiliary_pwp[3,C1,P1,CPU1] = 0
 con_auxiliary_pwp[3,C2,P1,CPU1]: - var_state_pw[3,C2,P1,CPU1]
   + var_action_sc[1,3,C2,P1,CPU1] + var_action_sc[2,3,C2,P1,CPU1]
   + var_action_sc[3,3,C2,P1,CPU1] + var_action_sc[4,3,C2,P1,CPU1]
   + var_auxiliary_pwp[3,C2,P1,CPU1] = 0
 con_auxiliary_pwp[4,C1,P1,CPU1]: - var_state_pw[4,C1,P1,CPU1]
   + var_action_sc[1,4,C1,P1,CPU1] + var_action_sc[2,4,C1,P1,CPU1]
   + var_action_sc[3,4,C1,P1,CPU1] + var_action_sc[4,4,C1,P1,CPU1]
   + var_auxiliary_pwp[4,C1,P1,CPU1] = 0
 con_auxiliary_pwp[4,C2,P1,CPU1]: - var_state_pw[4,C2,P1,CPU1]
   + var_action_sc[1,4,C2,P1,CPU1] + var_action_sc[2,4,C2,P1,CPU1]
   + var_action_sc[3,4,C2,P1,CPU1] + var_action_sc[4,4,C2,P1,CPU1]
   + var_auxiliary_pwp[4,C2,P1,CPU1] = 0
 con_auxiliary_pws[1,0,C1,P1,CPU1]: - var_state_ps[1,0,C1,P1,CPU1]
   - var_action_sc[1,0,C1,P1,CPU1] + var_action_rsc[1,2,0,C1,P1,CPU1]
   + var_action_rsc[1,3,0,C1,P1,CPU1] + var_action_rsc[1,4,0,C1,P1,CPU1]
   - var_action_rsc[2,1,0,C1,P1,CPU1] - var_action_rsc[3,1,0,C1,P1,CPU1]
   - var_action_rsc[4,1,0,C1,P1,CPU1] + var_auxiliary_psp[1,0,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[1,0,C2,P1,CPU1]: - var_state_ps[1,0,C2,P1,CPU1]
   - var_action_sc[1,0,C2,P1,CPU1] + var_action_rsc[1,2,0,C2,P1,CPU1]
   + var_action_rsc[1,3,0,C2,P1,CPU1] + var_action_rsc[1,4,0,C2,P1,CPU1]
   - var_action_rsc[2,1,0,C2,P1,CPU1] - var_action_rsc[3,1,0,C2,P1,CPU1]
   - var_action_rsc[4,1,0,C2,P1,CPU1] + var_auxiliary_psp[1,0,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[1,1,C1,P1,CPU1]: - var_state_ps[1,1,C1,P1,CPU1]
   - var_action_sc[1,1,C1,P1,CPU1] + var_action_rsc[1,2,1,C1,P1,CPU1]
   + var_action_rsc[1,3,1,C1,P1,CPU1] + var_action_rsc[1,4,1,C1,P1,CPU1]
   - var_action_rsc[2,1,1,C1,P1,CPU1] - var_action_rsc[3,1,1,C1,P1,CPU1]
   - var_action_rsc[4,1,1,C1,P1,CPU1] + var_auxiliary_psp[1,1,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[1,1,C2,P1,CPU1]: - var_state_ps[1,1,C2,P1,CPU1]
   - var_action_sc[1,1,C2,P1,CPU1] + var_action_rsc[1,2,1,C2,P1,CPU1]
   + var_action_rsc[1,3,1,C2,P1,CPU1] + var_action_rsc[1,4,1,C2,P1,CPU1]
   - var_action_rsc[2,1,1,C2,P1,CPU1] - var_action_rsc[3,1,1,C2,P1,CPU1]
   - var_action_rsc[4,1,1,C2,P1,CPU1] + var_auxiliary_psp[1,1,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[1,2,C1,P1,CPU1]: - var_state_ps[1,2,C1,P1,CPU1]
   - var_action_sc[1,2,C1,P1,CPU1] + var_action_rsc[1,2,2,C1,P1,CPU1]
   + var_action_rsc[1,3,2,C1,P1,CPU1] + var_action_rsc[1,4,2,C1,P1,CPU1]
   - var_action_rsc[2,1,2,C1,P1,CPU1] - var_action_rsc[3,1,2,C1,P1,CPU1]
   - var_action_rsc[4,1,2,C1,P1,CPU1] + var_auxiliary_psp[1,2,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[1,2,C2,P1,CPU1]: - var_state_ps[1,2,C2,P1,CPU1]
   - var_action_sc[1,2,C2,P1,CPU1] + var_action_rsc[1,2,2,C2,P1,CPU1]
   + var_action_rsc[1,3,2,C2,P1,CPU1] + var_action_rsc[1,4,2,C2,P1,CPU1]
   - var_action_rsc[2,1,2,C2,P1,CPU1] - var_action_rsc[3,1,2,C2,P1,CPU1]
   - var_action_rsc[4,1,2,C2,P1,CPU1] + var_auxiliary_psp[1,2,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[1,3,C1,P1,CPU1]: - var_state_ps[1,3,C1,P1,CPU1]
   - var_action_sc[1,3,C1,P1,CPU1] + var_action_rsc[1,2,3,C1,P1,CPU1]
   + var_action_rsc[1,3,3,C1,P1,CPU1] + var_action_rsc[1,4,3,C1,P1,CPU1]
   - var_action_rsc[2,1,3,C1,P1,CPU1] - var_action_rsc[3,1,3,C1,P1,CPU1]
   - var_action_rsc[4,1,3,C1,P1,CPU1] + var_auxiliary_psp[1,3,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[1,3,C2,P1,CPU1]: - var_state_ps[1,3,C2,P1,CPU1]
   - var_action_sc[1,3,C2,P1,CPU1] + var_action_rsc[1,2,3,C2,P1,CPU1]
   + var_action_rsc[1,3,3,C2,P1,CPU1] + var_action_rsc[1,4,3,C2,P1,CPU1]
   - var_action_rsc[2,1,3,C2,P1,CPU1] - var_action_rsc[3,1,3,C2,P1,CPU1]
   - var_action_rsc[4,1,3,C2,P1,CPU1] + var_auxiliary_psp[1,3,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[1,4,C1,P1,CPU1]: - var_state_ps[1,4,C1,P1,CPU1]
   - var_action_sc[1,4,C1,P1,CPU1] + var_action_rsc[1,2,4,C1,P1,CPU1]
   + var_action_rsc[1,3,4,C1,P1,CPU1] + var_action_rsc[1,4,4,C1,P1,CPU1]
   - var_action_rsc[2,1,4,C1,P1,CPU1] - var_action_rsc[3,1,4,C1,P1,CPU1]
   - var_action_rsc[4,1,4,C1,P1,CPU1] + var_auxiliary_psp[1,4,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[1,4,C2,P1,CPU1]: - var_state_ps[1,4,C2,P1,CPU1]
   - var_action_sc[1,4,C2,P1,CPU1] + var_action_rsc[1,2,4,C2,P1,CPU1]
   + var_action_rsc[1,3,4,C2,P1,CPU1] + var_action_rsc[1,4,4,C2,P1,CPU1]
   - var_action_rsc[2,1,4,C2,P1,CPU1] - var_action_rsc[3,1,4,C2,P1,CPU1]
   - var_action_rsc[4,1,4,C2,P1,CPU1] + var_auxiliary_psp[1,4,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[2,0,C1,P1,CPU1]: - var_state_ps[2,0,C1,P1,CPU1]
   - var_action_sc[2,0,C1,P1,CPU1] - var_action_rsc[1,2,0,C1,P1,CPU1]
   + var_action_rsc[2,1,0,C1,P1,CPU1] + var_action_rsc[2,3,0,C1,P1,CPU1]
   + var_action_rsc[2,4,0,C1,P1,CPU1] - var_action_rsc[3,2,0,C1,P1,CPU1]
   - var_action_rsc[4,2,0,C1,P1,CPU1] + var_auxiliary_psp[2,0,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[2,0,C2,P1,CPU1]: - var_state_ps[2,0,C2,P1,CPU1]
   - var_action_sc[2,0,C2,P1,CPU1] - var_action_rsc[1,2,0,C2,P1,CPU1]
   + var_action_rsc[2,1,0,C2,P1,CPU1] + var_action_rsc[2,3,0,C2,P1,CPU1]
   + var_action_rsc[2,4,0,C2,P1,CPU1] - var_action_rsc[3,2,0,C2,P1,CPU1]
   - var_action_rsc[4,2,0,C2,P1,CPU1] + var_auxiliary_psp[2,0,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[2,1,C1,P1,CPU1]: - var_state_ps[2,1,C1,P1,CPU1]
   - var_action_sc[2,1,C1,P1,CPU1] - var_action_rsc[1,2,1,C1,P1,CPU1]
   + var_action_rsc[2,1,1,C1,P1,CPU1] + var_action_rsc[2,3,1,C1,P1,CPU1]
   + var_action_rsc[2,4,1,C1,P1,CPU1] - var_action_rsc[3,2,1,C1,P1,CPU1]
   - var_action_rsc[4,2,1,C1,P1,CPU1] + var_auxiliary_psp[2,1,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[2,1,C2,P1,CPU1]: - var_state_ps[2,1,C2,P1,CPU1]
   - var_action_sc[2,1,C2,P1,CPU1] - var_action_rsc[1,2,1,C2,P1,CPU1]
   + var_action_rsc[2,1,1,C2,P1,CPU1] + var_action_rsc[2,3,1,C2,P1,CPU1]
   + var_action_rsc[2,4,1,C2,P1,CPU1] - var_action_rsc[3,2,1,C2,P1,CPU1]
   - var_action_rsc[4,2,1,C2,P1,CPU1] + var_auxiliary_psp[2,1,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[2,2,C1,P1,CPU1]: - var_state_ps[2,2,C1,P1,CPU1]
   - var_action_sc[2,2,C1,P1,CPU1] - var_action_rsc[1,2,2,C1,P1,CPU1]
   + var_action_rsc[2,1,2,C1,P1,CPU1] + var_action_rsc[2,3,2,C1,P1,CPU1]
   + var_action_rsc[2,4,2,C1,P1,CPU1] - var_action_rsc[3,2,2,C1,P1,CPU1]
   - var_action_rsc[4,2,2,C1,P1,CPU1] + var_auxiliary_psp[2,2,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[2,2,C2,P1,CPU1]: - var_state_ps[2,2,C2,P1,CPU1]
   - var_action_sc[2,2,C2,P1,CPU1] - var_action_rsc[1,2,2,C2,P1,CPU1]
   + var_action_rsc[2,1,2,C2,P1,CPU1] + var_action_rsc[2,3,2,C2,P1,CPU1]
   + var_action_rsc[2,4,2,C2,P1,CPU1] - var_action_rsc[3,2,2,C2,P1,CPU1]
   - var_action_rsc[4,2,2,C2,P1,CPU1] + var_auxiliary_psp[2,2,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[2,3,C1,P1,CPU1]: - var_state_ps[2,3,C1,P1,CPU1]
   - var_action_sc[2,3,C1,P1,CPU1] - var_action_rsc[1,2,3,C1,P1,CPU1]
   + var_action_rsc[2,1,3,C1,P1,CPU1] + var_action_rsc[2,3,3,C1,P1,CPU1]
   + var_action_rsc[2,4,3,C1,P1,CPU1] - var_action_rsc[3,2,3,C1,P1,CPU1]
   - var_action_rsc[4,2,3,C1,P1,CPU1] + var_auxiliary_psp[2,3,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[2,3,C2,P1,CPU1]: - var_state_ps[2,3,C2,P1,CPU1]
   - var_action_sc[2,3,C2,P1,CPU1] - var_action_rsc[1,2,3,C2,P1,CPU1]
   + var_action_rsc[2,1,3,C2,P1,CPU1] + var_action_rsc[2,3,3,C2,P1,CPU1]
   + var_action_rsc[2,4,3,C2,P1,CPU1] - var_action_rsc[3,2,3,C2,P1,CPU1]
   - var_action_rsc[4,2,3,C2,P1,CPU1] + var_auxiliary_psp[2,3,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[2,4,C1,P1,CPU1]: - var_state_ps[2,4,C1,P1,CPU1]
   - var_action_sc[2,4,C1,P1,CPU1] - var_action_rsc[1,2,4,C1,P1,CPU1]
   + var_action_rsc[2,1,4,C1,P1,CPU1] + var_action_rsc[2,3,4,C1,P1,CPU1]
   + var_action_rsc[2,4,4,C1,P1,CPU1] - var_action_rsc[3,2,4,C1,P1,CPU1]
   - var_action_rsc[4,2,4,C1,P1,CPU1] + var_auxiliary_psp[2,4,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[2,4,C2,P1,CPU1]: - var_state_ps[2,4,C2,P1,CPU1]
   - var_action_sc[2,4,C2,P1,CPU1] - var_action_rsc[1,2,4,C2,P1,CPU1]
   + var_action_rsc[2,1,4,C2,P1,CPU1] + var_action_rsc[2,3,4,C2,P1,CPU1]
   + var_action_rsc[2,4,4,C2,P1,CPU1] - var_action_rsc[3,2,4,C2,P1,CPU1]
   - var_action_rsc[4,2,4,C2,P1,CPU1] + var_auxiliary_psp[2,4,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[3,0,C1,P1,CPU1]: - var_state_ps[3,0,C1,P1,CPU1]
   - var_action_sc[3,0,C1,P1,CPU1] - var_action_rsc[1,3,0,C1,P1,CPU1]
   - var_action_rsc[2,3,0,C1,P1,CPU1] + var_action_rsc[3,1,0,C1,P1,CPU1]
   + var_action_rsc[3,2,0,C1,P1,CPU1] + var_action_rsc[3,4,0,C1,P1,CPU1]
   - var_action_rsc[4,3,0,C1,P1,CPU1] + var_auxiliary_psp[3,0,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[3,0,C2,P1,CPU1]: - var_state_ps[3,0,C2,P1,CPU1]
   - var_action_sc[3,0,C2,P1,CPU1] - var_action_rsc[1,3,0,C2,P1,CPU1]
   - var_action_rsc[2,3,0,C2,P1,CPU1] + var_action_rsc[3,1,0,C2,P1,CPU1]
   + var_action_rsc[3,2,0,C2,P1,CPU1] + var_action_rsc[3,4,0,C2,P1,CPU1]
   - var_action_rsc[4,3,0,C2,P1,CPU1] + var_auxiliary_psp[3,0,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[3,1,C1,P1,CPU1]: - var_state_ps[3,1,C1,P1,CPU1]
   - var_action_sc[3,1,C1,P1,CPU1] - var_action_rsc[1,3,1,C1,P1,CPU1]
   - var_action_rsc[2,3,1,C1,P1,CPU1] + var_action_rsc[3,1,1,C1,P1,CPU1]
   + var_action_rsc[3,2,1,C1,P1,CPU1] + var_action_rsc[3,4,1,C1,P1,CPU1]
   - var_action_rsc[4,3,1,C1,P1,CPU1] + var_auxiliary_psp[3,1,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[3,1,C2,P1,CPU1]: - var_state_ps[3,1,C2,P1,CPU1]
   - var_action_sc[3,1,C2,P1,CPU1] - var_action_rsc[1,3,1,C2,P1,CPU1]
   - var_action_rsc[2,3,1,C2,P1,CPU1] + var_action_rsc[3,1,1,C2,P1,CPU1]
   + var_action_rsc[3,2,1,C2,P1,CPU1] + var_action_rsc[3,4,1,C2,P1,CPU1]
   - var_action_rsc[4,3,1,C2,P1,CPU1] + var_auxiliary_psp[3,1,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[3,2,C1,P1,CPU1]: - var_state_ps[3,2,C1,P1,CPU1]
   - var_action_sc[3,2,C1,P1,CPU1] - var_action_rsc[1,3,2,C1,P1,CPU1]
   - var_action_rsc[2,3,2,C1,P1,CPU1] + var_action_rsc[3,1,2,C1,P1,CPU1]
   + var_action_rsc[3,2,2,C1,P1,CPU1] + var_action_rsc[3,4,2,C1,P1,CPU1]
   - var_action_rsc[4,3,2,C1,P1,CPU1] + var_auxiliary_psp[3,2,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[3,2,C2,P1,CPU1]: - var_state_ps[3,2,C2,P1,CPU1]
   - var_action_sc[3,2,C2,P1,CPU1] - var_action_rsc[1,3,2,C2,P1,CPU1]
   - var_action_rsc[2,3,2,C2,P1,CPU1] + var_action_rsc[3,1,2,C2,P1,CPU1]
   + var_action_rsc[3,2,2,C2,P1,CPU1] + var_action_rsc[3,4,2,C2,P1,CPU1]
   - var_action_rsc[4,3,2,C2,P1,CPU1] + var_auxiliary_psp[3,2,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[3,3,C1,P1,CPU1]: - var_state_ps[3,3,C1,P1,CPU1]
   - var_action_sc[3,3,C1,P1,CPU1] - var_action_rsc[1,3,3,C1,P1,CPU1]
   - var_action_rsc[2,3,3,C1,P1,CPU1] + var_action_rsc[3,1,3,C1,P1,CPU1]
   + var_action_rsc[3,2,3,C1,P1,CPU1] + var_action_rsc[3,4,3,C1,P1,CPU1]
   - var_action_rsc[4,3,3,C1,P1,CPU1] + var_auxiliary_psp[3,3,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[3,3,C2,P1,CPU1]: - var_state_ps[3,3,C2,P1,CPU1]
   - var_action_sc[3,3,C2,P1,CPU1] - var_action_rsc[1,3,3,C2,P1,CPU1]
   - var_action_rsc[2,3,3,C2,P1,CPU1] + var_action_rsc[3,1,3,C2,P1,CPU1]
   + var_action_rsc[3,2,3,C2,P1,CPU1] + var_action_rsc[3,4,3,C2,P1,CPU1]
   - var_action_rsc[4,3,3,C2,P1,CPU1] + var_auxiliary_psp[3,3,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[3,4,C1,P1,CPU1]: - var_state_ps[3,4,C1,P1,CPU1]
   - var_action_sc[3,4,C1,P1,CPU1] - var_action_rsc[1,3,4,C1,P1,CPU1]
   - var_action_rsc[2,3,4,C1,P1,CPU1] + var_action_rsc[3,1,4,C1,P1,CPU1]
   + var_action_rsc[3,2,4,C1,P1,CPU1] + var_action_rsc[3,4,4,C1,P1,CPU1]
   - var_action_rsc[4,3,4,C1,P1,CPU1] + var_auxiliary_psp[3,4,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[3,4,C2,P1,CPU1]: - var_state_ps[3,4,C2,P1,CPU1]
   - var_action_sc[3,4,C2,P1,CPU1] - var_action_rsc[1,3,4,C2,P1,CPU1]
   - var_action_rsc[2,3,4,C2,P1,CPU1] + var_action_rsc[3,1,4,C2,P1,CPU1]
   + var_action_rsc[3,2,4,C2,P1,CPU1] + var_action_rsc[3,4,4,C2,P1,CPU1]
   - var_action_rsc[4,3,4,C2,P1,CPU1] + var_auxiliary_psp[3,4,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[4,0,C1,P1,CPU1]: - var_state_ps[4,0,C1,P1,CPU1]
   - var_action_sc[4,0,C1,P1,CPU1] - var_action_rsc[1,4,0,C1,P1,CPU1]
   - var_action_rsc[2,4,0,C1,P1,CPU1] - var_action_rsc[3,4,0,C1,P1,CPU1]
   + var_action_rsc[4,1,0,C1,P1,CPU1] + var_action_rsc[4,2,0,C1,P1,CPU1]
   + var_action_rsc[4,3,0,C1,P1,CPU1] + var_auxiliary_psp[4,0,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[4,0,C2,P1,CPU1]: - var_state_ps[4,0,C2,P1,CPU1]
   - var_action_sc[4,0,C2,P1,CPU1] - var_action_rsc[1,4,0,C2,P1,CPU1]
   - var_action_rsc[2,4,0,C2,P1,CPU1] - var_action_rsc[3,4,0,C2,P1,CPU1]
   + var_action_rsc[4,1,0,C2,P1,CPU1] + var_action_rsc[4,2,0,C2,P1,CPU1]
   + var_action_rsc[4,3,0,C2,P1,CPU1] + var_auxiliary_psp[4,0,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[4,1,C1,P1,CPU1]: - var_state_ps[4,1,C1,P1,CPU1]
   - var_action_sc[4,1,C1,P1,CPU1] - var_action_rsc[1,4,1,C1,P1,CPU1]
   - var_action_rsc[2,4,1,C1,P1,CPU1] - var_action_rsc[3,4,1,C1,P1,CPU1]
   + var_action_rsc[4,1,1,C1,P1,CPU1] + var_action_rsc[4,2,1,C1,P1,CPU1]
   + var_action_rsc[4,3,1,C1,P1,CPU1] + var_auxiliary_psp[4,1,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[4,1,C2,P1,CPU1]: - var_state_ps[4,1,C2,P1,CPU1]
   - var_action_sc[4,1,C2,P1,CPU1] - var_action_rsc[1,4,1,C2,P1,CPU1]
   - var_action_rsc[2,4,1,C2,P1,CPU1] - var_action_rsc[3,4,1,C2,P1,CPU1]
   + var_action_rsc[4,1,1,C2,P1,CPU1] + var_action_rsc[4,2,1,C2,P1,CPU1]
   + var_action_rsc[4,3,1,C2,P1,CPU1] + var_auxiliary_psp[4,1,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[4,2,C1,P1,CPU1]: - var_state_ps[4,2,C1,P1,CPU1]
   - var_action_sc[4,2,C1,P1,CPU1] - var_action_rsc[1,4,2,C1,P1,CPU1]
   - var_action_rsc[2,4,2,C1,P1,CPU1] - var_action_rsc[3,4,2,C1,P1,CPU1]
   + var_action_rsc[4,1,2,C1,P1,CPU1] + var_action_rsc[4,2,2,C1,P1,CPU1]
   + var_action_rsc[4,3,2,C1,P1,CPU1] + var_auxiliary_psp[4,2,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[4,2,C2,P1,CPU1]: - var_state_ps[4,2,C2,P1,CPU1]
   - var_action_sc[4,2,C2,P1,CPU1] - var_action_rsc[1,4,2,C2,P1,CPU1]
   - var_action_rsc[2,4,2,C2,P1,CPU1] - var_action_rsc[3,4,2,C2,P1,CPU1]
   + var_action_rsc[4,1,2,C2,P1,CPU1] + var_action_rsc[4,2,2,C2,P1,CPU1]
   + var_action_rsc[4,3,2,C2,P1,CPU1] + var_auxiliary_psp[4,2,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[4,3,C1,P1,CPU1]: - var_state_ps[4,3,C1,P1,CPU1]
   - var_action_sc[4,3,C1,P1,CPU1] - var_action_rsc[1,4,3,C1,P1,CPU1]
   - var_action_rsc[2,4,3,C1,P1,CPU1] - var_action_rsc[3,4,3,C1,P1,CPU1]
   + var_action_rsc[4,1,3,C1,P1,CPU1] + var_action_rsc[4,2,3,C1,P1,CPU1]
   + var_action_rsc[4,3,3,C1,P1,CPU1] + var_auxiliary_psp[4,3,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[4,3,C2,P1,CPU1]: - var_state_ps[4,3,C2,P1,CPU1]
   - var_action_sc[4,3,C2,P1,CPU1] - var_action_rsc[1,4,3,C2,P1,CPU1]
   - var_action_rsc[2,4,3,C2,P1,CPU1] - var_action_rsc[3,4,3,C2,P1,CPU1]
   + var_action_rsc[4,1,3,C2,P1,CPU1] + var_action_rsc[4,2,3,C2,P1,CPU1]
   + var_action_rsc[4,3,3,C2,P1,CPU1] + var_auxiliary_psp[4,3,C2,P1,CPU1]
   = 0
 con_auxiliary_pws[4,4,C1,P1,CPU1]: - var_state_ps[4,4,C1,P1,CPU1]
   - var_action_sc[4,4,C1,P1,CPU1] - var_action_rsc[1,4,4,C1,P1,CPU1]
   - var_action_rsc[2,4,4,C1,P1,CPU1] - var_action_rsc[3,4,4,C1,P1,CPU1]
   + var_action_rsc[4,1,4,C1,P1,CPU1] + var_action_rsc[4,2,4,C1,P1,CPU1]
   + var_action_rsc[4,3,4,C1,P1,CPU1] + var_auxiliary_psp[4,4,C1,P1,CPU1]
   = 0
 con_auxiliary_pws[4,4,C2,P1,CPU1]: - var_state_ps[4,4,C2,P1,CPU1]
   - var_action_sc[4,4,C2,P1,CPU1] - var_action_rsc[1,4,4,C2,P1,CPU1]
   - var_action_rsc[2,4,4,C2,P1,CPU1] - var_action_rsc[3,4,4,C2,P1,CPU1]
   + var_action_rsc[4,1,4,C2,P1,CPU1] + var_action_rsc[4,2,4,C2,P1,CPU1]
   + var_action_rsc[4,3,4,C2,P1,CPU1] + var_auxiliary_psp[4,4,C2,P1,CPU1]
   = 0
 con_auxiliary_uv_0[1,PPE1]: var_auxiliary_uv[1,PPE1] >= 0
 con_auxiliary_uv_0[2,PPE1]: var_auxiliary_uv[2,PPE1] >= 0
 con_auxiliary_uv_0[3,PPE1]: var_auxiliary_uv[3,PPE1] >= 0
 con_auxiliary_uv_0[4,PPE1]: var_auxiliary_uv[4,PPE1] >= 0
 con_auxiliary_uv_0M[1,PPE1]: var_auxiliary_uv[1,PPE1]
   - 10000 var_auxiliary_uvb[1,PPE1] <= 0
 con_auxiliary_uv_0M[2,PPE1]: var_auxiliary_uv[2,PPE1]
   - 10000 var_auxiliary_uvb[2,PPE1] <= 0
 con_auxiliary_uv_0M[3,PPE1]: var_auxiliary_uv[3,PPE1]
   - 10000 var_auxiliary_uvb[3,PPE1] <= 0
 con_auxiliary_uv_0M[4,PPE1]: var_auxiliary_uv[4,PPE1]
   - 10000 var_auxiliary_uvb[4,PPE1] <= 0
 con_auxiliary_uv_1[PPE1]: var_state_ul[PPE1] + var_auxiliary_uv[1,PPE1]
   - var_auxiliary_uup[1,PPE1] >= -30
 con_auxiliary_uv_1M[PPE1]: var_state_ul[PPE1] + var_auxiliary_uv[1,PPE1]
   + 10000 var_auxiliary_uvb[1,PPE1] - var_auxiliary_uup[1,PPE1] <= 9970
 con_auxiliary_uv_m[1,PPE1]: var_auxiliary_uv[1,PPE1]
   - var_auxiliary_uup[1,PPE1] >= -30
 con_auxiliary_uv_m[2,PPE1]: var_auxiliary_uv[2,PPE1]
   - var_auxiliary_uup[2,PPE1] >= -30
 con_auxiliary_uv_m[3,PPE1]: var_auxiliary_uv[3,PPE1]
   - var_auxiliary_uup[3,PPE1] >= -30
 con_auxiliary_uv_m[4,PPE1]: var_auxiliary_uv[4,PPE1]
   - var_auxiliary_uup[4,PPE1] >= -30
 con_auxiliary_uv_mM[1,PPE1]: var_auxiliary_uv[1,PPE1]
   + 10000 var_auxiliary_uvb[1,PPE1] - var_auxiliary_uup[1,PPE1] <= 9970
 con_auxiliary_uv_mM[2,PPE1]: var_auxiliary_uv[2,PPE1]
   + 10000 var_auxiliary_uvb[2,PPE1] - var_auxiliary_uup[2,PPE1] <= 9970
 con_auxiliary_uv_mM[3,PPE1]: var_auxiliary_uv[3,PPE1]
   + 10000 var_auxiliary_uvb[3,PPE1] - var_auxiliary_uup[3,PPE1] <= 9970
 con_auxiliary_uv_mM[4,PPE1]: var_auxiliary_uv[4,PPE1]
   + 10000 var_auxiliary_uvb[4,PPE1] - var_auxiliary_uup[4,PPE1] <= 9970
 con_usage_1[PPE1]: - var_state_ul[PPE1] - var_auxiliary_uv[1,PPE1]
   + var_auxiliary_uup[1,PPE1] <= 30
 con_usage_tT[2,PPE1]: - var_auxiliary_uv[2,PPE1]
   + var_auxiliary_uup[2,PPE1] <= 30
 con_usage_tT[3,PPE1]: - var_auxiliary_uv[3,PPE1]
   + var_auxiliary_uup[3,PPE1] <= 30
 con_usage_tT[4,PPE1]: - var_auxiliary_uv[4,PPE1]
   + var_auxiliary_uup[4,PPE1] <= 30
 con_reschedule_bounds[2,2,0,C1,P1,CPU1]: var_action_rsc[2,2,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,0,C2,P1,CPU1]: var_action_rsc[2,2,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,1,C1,P1,CPU1]: var_action_rsc[2,2,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,1,C2,P1,CPU1]: var_action_rsc[2,2,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,2,C1,P1,CPU1]: var_action_rsc[2,2,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,2,C2,P1,CPU1]: var_action_rsc[2,2,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,3,C1,P1,CPU1]: var_action_rsc[2,2,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,3,C2,P1,CPU1]: var_action_rsc[2,2,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,4,C1,P1,CPU1]: var_action_rsc[2,2,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,2,4,C2,P1,CPU1]: var_action_rsc[2,2,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,0,C1,P1,CPU1]: var_action_rsc[2,3,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,0,C2,P1,CPU1]: var_action_rsc[2,3,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,1,C1,P1,CPU1]: var_action_rsc[2,3,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,1,C2,P1,CPU1]: var_action_rsc[2,3,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,2,C1,P1,CPU1]: var_action_rsc[2,3,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,2,C2,P1,CPU1]: var_action_rsc[2,3,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,3,C1,P1,CPU1]: var_action_rsc[2,3,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,3,C2,P1,CPU1]: var_action_rsc[2,3,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,4,C1,P1,CPU1]: var_action_rsc[2,3,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,3,4,C2,P1,CPU1]: var_action_rsc[2,3,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,0,C1,P1,CPU1]: var_action_rsc[2,4,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,0,C2,P1,CPU1]: var_action_rsc[2,4,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,1,C1,P1,CPU1]: var_action_rsc[2,4,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,1,C2,P1,CPU1]: var_action_rsc[2,4,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,2,C1,P1,CPU1]: var_action_rsc[2,4,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,2,C2,P1,CPU1]: var_action_rsc[2,4,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,3,C1,P1,CPU1]: var_action_rsc[2,4,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,3,C2,P1,CPU1]: var_action_rsc[2,4,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,4,C1,P1,CPU1]: var_action_rsc[2,4,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[2,4,4,C2,P1,CPU1]: var_action_rsc[2,4,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,0,C1,P1,CPU1]: var_action_rsc[3,2,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,0,C2,P1,CPU1]: var_action_rsc[3,2,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,1,C1,P1,CPU1]: var_action_rsc[3,2,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,1,C2,P1,CPU1]: var_action_rsc[3,2,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,2,C1,P1,CPU1]: var_action_rsc[3,2,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,2,C2,P1,CPU1]: var_action_rsc[3,2,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,3,C1,P1,CPU1]: var_action_rsc[3,2,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,3,C2,P1,CPU1]: var_action_rsc[3,2,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,4,C1,P1,CPU1]: var_action_rsc[3,2,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,2,4,C2,P1,CPU1]: var_action_rsc[3,2,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,0,C1,P1,CPU1]: var_action_rsc[3,3,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,0,C2,P1,CPU1]: var_action_rsc[3,3,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,1,C1,P1,CPU1]: var_action_rsc[3,3,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,1,C2,P1,CPU1]: var_action_rsc[3,3,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,2,C1,P1,CPU1]: var_action_rsc[3,3,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,2,C2,P1,CPU1]: var_action_rsc[3,3,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,3,C1,P1,CPU1]: var_action_rsc[3,3,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,3,C2,P1,CPU1]: var_action_rsc[3,3,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,4,C1,P1,CPU1]: var_action_rsc[3,3,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,3,4,C2,P1,CPU1]: var_action_rsc[3,3,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,0,C1,P1,CPU1]: var_action_rsc[3,4,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,0,C2,P1,CPU1]: var_action_rsc[3,4,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,1,C1,P1,CPU1]: var_action_rsc[3,4,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,1,C2,P1,CPU1]: var_action_rsc[3,4,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,2,C1,P1,CPU1]: var_action_rsc[3,4,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,2,C2,P1,CPU1]: var_action_rsc[3,4,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,3,C1,P1,CPU1]: var_action_rsc[3,4,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,3,C2,P1,CPU1]: var_action_rsc[3,4,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,4,C1,P1,CPU1]: var_action_rsc[3,4,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[3,4,4,C2,P1,CPU1]: var_action_rsc[3,4,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,0,C1,P1,CPU1]: var_action_rsc[4,2,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,0,C2,P1,CPU1]: var_action_rsc[4,2,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,1,C1,P1,CPU1]: var_action_rsc[4,2,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,1,C2,P1,CPU1]: var_action_rsc[4,2,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,2,C1,P1,CPU1]: var_action_rsc[4,2,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,2,C2,P1,CPU1]: var_action_rsc[4,2,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,3,C1,P1,CPU1]: var_action_rsc[4,2,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,3,C2,P1,CPU1]: var_action_rsc[4,2,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,4,C1,P1,CPU1]: var_action_rsc[4,2,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,2,4,C2,P1,CPU1]: var_action_rsc[4,2,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,0,C1,P1,CPU1]: var_action_rsc[4,3,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,0,C2,P1,CPU1]: var_action_rsc[4,3,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,1,C1,P1,CPU1]: var_action_rsc[4,3,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,1,C2,P1,CPU1]: var_action_rsc[4,3,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,2,C1,P1,CPU1]: var_action_rsc[4,3,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,2,C2,P1,CPU1]: var_action_rsc[4,3,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,3,C1,P1,CPU1]: var_action_rsc[4,3,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,3,C2,P1,CPU1]: var_action_rsc[4,3,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,4,C1,P1,CPU1]: var_action_rsc[4,3,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,3,4,C2,P1,CPU1]: var_action_rsc[4,3,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,0,C1,P1,CPU1]: var_action_rsc[4,4,0,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,0,C2,P1,CPU1]: var_action_rsc[4,4,0,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,1,C1,P1,CPU1]: var_action_rsc[4,4,1,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,1,C2,P1,CPU1]: var_action_rsc[4,4,1,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,2,C1,P1,CPU1]: var_action_rsc[4,4,2,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,2,C2,P1,CPU1]: var_action_rsc[4,4,2,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,3,C1,P1,CPU1]: var_action_rsc[4,4,3,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,3,C2,P1,CPU1]: var_action_rsc[4,4,3,C2,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,4,C1,P1,CPU1]: var_action_rsc[4,4,4,C1,P1,CPU1]
   = 0
 con_reschedule_bounds[4,4,4,C2,P1,CPU1]: var_action_rsc[4,4,4,C2,P1,CPU1]
   = 0
 con_reschedule_bounds_ttp[1,1,0,C1,P1,CPU1]:
   var_action_rsc[1,1,0,C1,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,0,C2,P1,CPU1]:
   var_action_rsc[1,1,0,C2,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,1,C1,P1,CPU1]:
   var_action_rsc[1,1,1,C1,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,1,C2,P1,CPU1]:
   var_action_rsc[1,1,1,C2,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,2,C1,P1,CPU1]:
   var_action_rsc[1,1,2,C1,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,2,C2,P1,CPU1]:
   var_action_rsc[1,1,2,C2,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,3,C1,P1,CPU1]:
   var_action_rsc[1,1,3,C1,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,3,C2,P1,CPU1]:
   var_action_rsc[1,1,3,C2,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,4,C1,P1,CPU1]:
   var_action_rsc[1,1,4,C1,P1,CPU1] = 0
 con_reschedule_bounds_ttp[1,1,4,C2,P1,CPU1]:
   var_action_rsc[1,1,4,C2,P1,CPU1] = 0
Bounds
Binaries
 var_auxiliary_uvb[1,PPE1] var_auxiliary_uvb[2,PPE1]
 var_auxiliary_uvb[3,PPE1] var_auxiliary_uvb[4,PPE1]
Generals
 var_state_pw[0,C1,P1,CPU1] var_state_pw[0,C2,P1,CPU1]
 var_state_pw[1,C1,P1,CPU1] var_state_pw[1,C2,P1,CPU1]
 var_state_pw[2,C1,P1,CPU1] var_state_pw[2,C2,P1,CPU1]
 var_state_pw[3,C1,P1,CPU1] var_state_pw[3,C2,P1,CPU1]
 var_state_pw[4,C1,P1,CPU1] var_state_pw[4,C2,P1,CPU1]
 var_state_ps[1,0,C1,P1,CPU1] var_state_ps[1,0,C2,P1,CPU1]
 var_state_ps[1,1,C1,P1,CPU1] var_state_ps[1,1,C2,P1,CPU1]
 var_state_ps[1,2,C1,P1,CPU1] var_state_ps[1,2,C2,P1,CPU1]
 var_state_ps[1,3,C1,P1,CPU1] var_state_ps[1,3,C2,P1,CPU1]
 var_state_ps[1,4,C1,P1,CPU1] var_state_ps[1,4,C2,P1,CPU1]
 var_state_ps[2,0,C1,P1,CPU1] var_state_ps[2,0,C2,P1,CPU1]
 var_state_ps[2,1,C1,P1,CPU1] var_state_ps[2,1,C2,P1,CPU1]
 var_state_ps[2,2,C1,P1,CPU1] var_state_ps[2,2,C2,P1,CPU1]
 var_state_ps[2,3,C1,P1,CPU1] var_state_ps[2,3,C2,P1,CPU1]
 var_state_ps[2,4,C1,P1,CPU1] var_state_ps[2,4,C2,P1,CPU1]
 var_state_ps[3,0,C1,P1,CPU1] var_state_ps[3,0,C2,P1,CPU1]
 var_state_ps[3,1,C1,P1,CPU1] var_state_ps[3,1,C2,P1,CPU1]
 var_state_ps[3,2,C1,P1,CPU1] var_state_ps[3,2,C2,P1,CPU1]
 var_state_ps[3,3,C1,P1,CPU1] var_state_ps[3,3,C2,P1,CPU1]
 var_state_ps[3,4,C1,P1,CPU1] var_state_ps[3,4,C2,P1,CPU1]
 var_state_ps[4,0,C1,P1,CPU1] var_state_ps[4,0,C2,P1,CPU1]
 var_state_ps[4,1,C1,P1,CPU1] var_state_ps[4,1,C2,P1,CPU1]
 var_state_ps[4,2,C1,P1,CPU1] var_state_ps[4,2,C2,P1,CPU1]
 var_state_ps[4,3,C1,P1,CPU1] var_state_ps[4,3,C2,P1,CPU1]
 var_state_ps[4,4,C1,P1,CPU1] var_state_ps[4,4,C2,P1,CPU1]
 var_action_sc[1,0,C1,P1,CPU1] var_action_sc[1,0,C2,P1,CPU1]
 var_action_sc[1,1,C1,P1,CPU1] var_action_sc[1,1,C2,P1,CPU1]
 var_action_sc[1,2,C1,P1,CPU1] var_action_sc[1,2,C2,P1,CPU1]
 var_action_sc[1,3,C1,P1,CPU1] var_action_sc[1,3,C2,P1,CPU1]
 var_action_sc[1,4,C1,P1,CPU1] var_action_sc[1,4,C2,P1,CPU1]
 var_action_sc[2,0,C1,P1,CPU1] var_action_sc[2,0,C2,P1,CPU1]
 var_action_sc[2,1,C1,P1,CPU1] var_action_sc[2,1,C2,P1,CPU1]
 var_action_sc[2,2,C1,P1,CPU1] var_action_sc[2,2,C2,P1,CPU1]
 var_action_sc[2,3,C1,P1,CPU1] var_action_sc[2,3,C2,P1,CPU1]
 var_action_sc[2,4,C1,P1,CPU1] var_action_sc[2,4,C2,P1,CPU1]
 var_action_sc[3,0,C1,P1,CPU1] var_action_sc[3,0,C2,P1,CPU1]
 var_action_sc[3,1,C1,P1,CPU1] var_action_sc[3,1,C2,P1,CPU1]
 var_action_sc[3,2,C1,P1,CPU1] var_action_sc[3,2,C2,P1,CPU1]
 var_action_sc[3,3,C1,P1,CPU1] var_action_sc[3,3,C2,P1,CPU1]
 var_action_sc[3,4,C1,P1,CPU1] var_action_sc[3,4,C2,P1,CPU1]
 var_action_sc[4,0,C1,P1,CPU1] var_action_sc[4,0,C2,P1,CPU1]
 var_action_sc[4,1,C1,P1,CPU1] var_action_sc[4,1,C2,P1,CPU1]
 var_action_sc[4,2,C1,P1,CPU1] var_action_sc[4,2,C2,P1,CPU1]
 var_action_sc[4,3,C1,P1,CPU1] var_action_sc[4,3,C2,P1,CPU1]
 var_action_sc[4,4,C1,P1,CPU1] var_action_sc[4,4,C2,P1,CPU1]
 var_action_rsc[1,1,0,C1,P1,CPU1] var_action_rsc[1,1,0,C2,P1,CPU1]
 var_action_rsc[1,1,1,C1,P1,CPU1] var_action_rsc[1,1,1,C2,P1,CPU1]
 var_action_rsc[1,1,2,C1,P1,CPU1] var_action_rsc[1,1,2,C2,P1,CPU1]
 var_action_rsc[1,1,3,C1,P1,CPU1] var_action_rsc[1,1,3,C2,P1,CPU1]
 var_action_rsc[1,1,4,C1,P1,CPU1] var_action_rsc[1,1,4,C2,P1,CPU1]
 var_action_rsc[1,2,0,C1,P1,CPU1] var_action_rsc[1,2,0,C2,P1,CPU1]
 var_action_rsc[1,2,1,C1,P1,CPU1] var_action_rsc[1,2,1,C2,P1,CPU1]
 var_action_rsc[1,2,2,C1,P1,CPU1] var_action_rsc[1,2,2,C2,P1,CPU1]
 var_action_rsc[1,2,3,C1,P1,CPU1] var_action_rsc[1,2,3,C2,P1,CPU1]
 var_action_rsc[1,2,4,C1,P1,CPU1] var_action_rsc[1,2,4,C2,P1,CPU1]
 var_action_rsc[1,3,0,C1,P1,CPU1] var_action_rsc[1,3,0,C2,P1,CPU1]
 var_action_rsc[1,3,1,C1,P1,CPU1] var_action_rsc[1,3,1,C2,P1,CPU1]
 var_action_rsc[1,3,2,C1,P1,CPU1] var_action_rsc[1,3,2,C2,P1,CPU1]
 var_action_rsc[1,3,3,C1,P1,CPU1] var_action_rsc[1,3,3,C2,P1,CPU1]
 var_action_rsc[1,3,4,C1,P1,CPU1] var_action_rsc[1,3,4,C2,P1,CPU1]
 var_action_rsc[1,4,0,C1,P1,CPU1] var_action_rsc[1,4,0,C2,P1,CPU1]
 var_action_rsc[1,4,1,C1,P1,CPU1] var_action_rsc[1,4,1,C2,P1,CPU1]
 var_action_rsc[1,4,2,C1,P1,CPU1] var_action_rsc[1,4,2,C2,P1,CPU1]
 var_action_rsc[1,4,3,C1,P1,CPU1] var_action_rsc[1,4,3,C2,P1,CPU1]
 var_action_rsc[1,4,4,C1,P1,CPU1] var_action_rsc[1,4,4,C2,P1,CPU1]
 var_action_rsc[2,1,0,C1,P1,CPU1] var_action_rsc[2,1,0,C2,P1,CPU1]
 var_action_rsc[2,1,1,C1,P1,CPU1] var_action_rsc[2,1,1,C2,P1,CPU1]
 var_action_rsc[2,1,2,C1,P1,CPU1] var_action_rsc[2,1,2,C2,P1,CPU1]
 var_action_rsc[2,1,3,C1,P1,CPU1] var_action_rsc[2,1,3,C2,P1,CPU1]
 var_action_rsc[2,1,4,C1,P1,CPU1] var_action_rsc[2,1,4,C2,P1,CPU1]
 var_action_rsc[2,2,0,C1,P1,CPU1] var_action_rsc[2,2,0,C2,P1,CPU1]
 var_action_rsc[2,2,1,C1,P1,CPU1] var_action_rsc[2,2,1,C2,P1,CPU1]
 var_action_rsc[2,2,2,C1,P1,CPU1] var_action_rsc[2,2,2,C2,P1,CPU1]
 var_action_rsc[2,2,3,C1,P1,CPU1] var_action_rsc[2,2,3,C2,P1,CPU1]
 var_action_rsc[2,2,4,C1,P1,CPU1] var_action_rsc[2,2,4,C2,P1,CPU1]
 var_action_rsc[2,3,0,C1,P1,CPU1] var_action_rsc[2,3,0,C2,P1,CPU1]
 var_action_rsc[2,3,1,C1,P1,CPU1] var_action_rsc[2,3,1,C2,P1,CPU1]
 var_action_rsc[2,3,2,C1,P1,CPU1] var_action_rsc[2,3,2,C2,P1,CPU1]
 var_action_rsc[2,3,3,C1,P1,CPU1] var_action_rsc[2,3,3,C2,P1,CPU1]
 var_action_rsc[2,3,4,C1,P1,CPU1] var_action_rsc[2,3,4,C2,P1,CPU1]
 var_action_rsc[2,4,0,C1,P1,CPU1] var_action_rsc[2,4,0,C2,P1,CPU1]
 var_action_rsc[2,4,1,C1,P1,CPU1] var_action_rsc[2,4,1,C2,P1,CPU1]
 var_action_rsc[2,4,2,C1,P1,CPU1] var_action_rsc[2,4,2,C2,P1,CPU1]
 var_action_rsc[2,4,3,C1,P1,CPU1] var_action_rsc[2,4,3,C2,P1,CPU1]
 var_action_rsc[2,4,4,C1,P1,CPU1] var_action_rsc[2,4,4,C2,P1,CPU1]
 var_action_rsc[3,1,0,C1,P1,CPU1] var_action_rsc[3,1,0,C2,P1,CPU1]
 var_action_rsc[3,1,1,C1,P1,CPU1] var_action_rsc[3,1,1,C2,P1,CPU1]
 var_action_rsc[3,1,2,C1,P1,CPU1] var_action_rsc[3,1,2,C2,P1,CPU1]
 var_action_rsc[3,1,3,C1,P1,CPU1] var_action_rsc[3,1,3,C2,P1,CPU1]
 var_action_rsc[3,1,4,C1,P1,CPU1] var_action_rsc[3,1,4,C2,P1,CPU1]
 var_action_rsc[3,2,0,C1,P1,CPU1] var_action_rsc[3,2,0,C2,P1,CPU1]
 var_action_rsc[3,2,1,C1,P1,CPU1] var_action_rsc[3,2,1,C2,P1,CPU1]
 var_action_rsc[3,2,2,C1,P1,CPU1] var_action_rsc[3,2,2,C2,P1,CPU1]
 var_action_rsc[3,2,3,C1,P1,CPU1] var_action_rsc[3,2,3,C2,P1,CPU1]
 var_action_rsc[3,2,4,C1,P1,CPU1] var_action_rsc[3,2,4,C2,P1,CPU1]
 var_action_rsc[3,3,0,C1,P1,CPU1] var_action_rsc[3,3,0,C2,P1,CPU1]
 var_action_rsc[3,3,1,C1,P1,CPU1] var_action_rsc[3,3,1,C2,P1,CPU1]
 var_action_rsc[3,3,2,C1,P1,CPU1] var_action_rsc[3,3,2,C2,P1,CPU1]
 var_action_rsc[3,3,3,C1,P1,CPU1] var_action_rsc[3,3,3,C2,P1,CPU1]
 var_action_rsc[3,3,4,C1,P1,CPU1] var_action_rsc[3,3,4,C2,P1,CPU1]
 var_action_rsc[3,4,0,C1,P1,CPU1] var_action_rsc[3,4,0,C2,P1,CPU1]
 var_action_rsc[3,4,1,C1,P1,CPU1] var_action_rsc[3,4,1,C2,P1,CPU1]
 var_action_rsc[3,4,2,C1,P1,CPU1] var_action_rsc[3,4,2,C2,P1,CPU1]
 var_action_rsc[3,4,3,C1,P1,CPU1] var_action_rsc[3,4,3,C2,P1,CPU1]
 var_action_rsc[3,4,4,C1,P1,CPU1] var_action_rsc[3,4,4,C2,P1,CPU1]
 var_action_rsc[4,1,0,C1,P1,CPU1] var_action_rsc[4,1,0,C2,P1,CPU1]
 var_action_rsc[4,1,1,C1,P1,CPU1] var_action_rsc[4,1,1,C2,P1,CPU1]
 var_action_rsc[4,1,2,C1,P1,CPU1] var_action_rsc[4,1,2,C2,P1,CPU1]
 var_action_rsc[4,1,3,C1,P1,CPU1] var_action_rsc[4,1,3,C2,P1,CPU1]
 var_action_rsc[4,1,4,C1,P1,CPU1] var_action_rsc[4,1,4,C2,P1,CPU1]
 var_action_rsc[4,2,0,C1,P1,CPU1] var_action_rsc[4,2,0,C2,P1,CPU1]
 var_action_rsc[4,2,1,C1,P1,CPU1] var_action_rsc[4,2,1,C2,P1,CPU1]
 var_action_rsc[4,2,2,C1,P1,CPU1] var_action_rsc[4,2,2,C2,P1,CPU1]
 var_action_rsc[4,2,3,C1,P1,CPU1] var_action_rsc[4,2,3,C2,P1,CPU1]
 var_action_rsc[4,2,4,C1,P1,CPU1] var_action_rsc[4,2,4,C2,P1,CPU1]
 var_action_rsc[4,3,0,C1,P1,CPU1] var_action_rsc[4,3,0,C2,P1,CPU1]
 var_action_rsc[4,3,1,C1,P1,CPU1] var_action_rsc[4,3,1,C2,P1,CPU1]
 var_action_rsc[4,3,2,C1,P1,CPU1] var_action_rsc[4,3,2,C2,P1,CPU1]
 var_action_rsc[4,3,3,C1,P1,CPU1] var_action_rsc[4,3,3,C2,P1,CPU1]
 var_action_rsc[4,3,4,C1,P1,CPU1] var_action_rsc[4,3,4,C2,P1,CPU1]
 var_action_rsc[4,4,0,C1,P1,CPU1] var_action_rsc[4,4,0,C2,P1,CPU1]
 var_action_rsc[4,4,1,C1,P1,CPU1] var_action_rsc[4,4,1,C2,P1,CPU1]
 var_action_rsc[4,4,2,C1,P1,CPU1] var_action_rsc[4,4,2,C2,P1,CPU1]
 var_action_rsc[4,4,3,C1,P1,CPU1] var_action_rsc[4,4,3,C2,P1,CPU1]
 var_action_rsc[4,4,4,C1,P1,CPU1] var_action_rsc[4,4,4,C2,P1,CPU1]
 var_auxiliary_pwp[0,C1,P1,CPU1] var_auxiliary_pwp[0,C2,P1,CPU1]
 var_auxiliary_pwp[1,C1,P1,CPU1] var_auxiliary_pwp[1,C2,P1,CPU1]
 var_auxiliary_pwp[2,C1,P1,CPU1] var_auxiliary_pwp[2,C2,P1,CPU1]
 var_auxiliary_pwp[3,C1,P1,CPU1] var_auxiliary_pwp[3,C2,P1,CPU1]
 var_auxiliary_pwp[4,C1,P1,CPU1] var_auxiliary_pwp[4,C2,P1,CPU1]
 var_auxiliary_psp[1,0,C1,P1,CPU1] var_auxiliary_psp[1,0,C2,P1,CPU1]
 var_auxiliary_psp[1,1,C1,P1,CPU1] var_auxiliary_psp[1,1,C2,P1,CPU1]
 var_auxiliary_psp[1,2,C1,P1,CPU1] var_auxiliary_psp[1,2,C2,P1,CPU1]
 var_auxiliary_psp[1,3,C1,P1,CPU1] var_auxiliary_psp[1,3,C2,P1,CPU1]
 var_auxiliary_psp[1,4,C1,P1,CPU1] var_auxiliary_psp[1,4,C2,P1,CPU1]
 var_auxiliary_psp[2,0,C1,P1,CPU1] var_auxiliary_psp[2,0,C2,P1,CPU1]
 var_auxiliary_psp[2,1,C1,P1,CPU1] var_auxiliary_psp[2,1,C2,P1,CPU1]
 var_auxiliary_psp[2,2,C1,P1,CPU1] var_auxiliary_psp[2,2,C2,P1,CPU1]
 var_auxiliary_psp[2,3,C1,P1,CPU1] var_auxiliary_psp[2,3,C2,P1,CPU1]
 var_auxiliary_psp[2,4,C1,P1,CPU1] var_auxiliary_psp[2,4,C2,P1,CPU1]
 var_auxiliary_psp[3,0,C1,P1,CPU1] var_auxiliary_psp[3,0,C2,P1,CPU1]
 var_auxiliary_psp[3,1,C1,P1,CPU1] var_auxiliary_psp[3,1,C2,P1,CPU1]
 var_auxiliary_psp[3,2,C1,P1,CPU1] var_auxiliary_psp[3,2,C2,P1,CPU1]
 var_auxiliary_psp[3,3,C1,P1,CPU1] var_auxiliary_psp[3,3,C2,P1,CPU1]
 var_auxiliary_psp[3,4,C1,P1,CPU1] var_auxiliary_psp[3,4,C2,P1,CPU1]
 var_auxiliary_psp[4,0,C1,P1,CPU1] var_auxiliary_psp[4,0,C2,P1,CPU1]
 var_auxiliary_psp[4,1,C1,P1,CPU1] var_auxiliary_psp[4,1,C2,P1,CPU1]
 var_auxiliary_psp[4,2,C1,P1,CPU1] var_auxiliary_psp[4,2,C2,P1,CPU1]
 var_auxiliary_psp[4,3,C1,P1,CPU1] var_auxiliary_psp[4,3,C2,P1,CPU1]
 var_auxiliary_psp[4,4,C1,P1,CPU1] var_auxiliary_psp[4,4,C2,P1,CPU1]
End
