\ Model MasterKnapsack_copy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  gv_b_0 + gv_b_ul_('PPE1',) + gv_b_pw_(0,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(1,_'C1',_'P1',_'CPU1') + gv_b_pw_(2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_2,_'C1',_'P1',_'CPU1')
Subject To
 b_0: 0.01 sa_0 + 0.01 sa_1 + 0.01 sa_2 + 0.01 sa_3 + 0.01 sa_4 + gv_b_0
   = 1
 b_ul_'PPE1': 60 sa_1 + gv_b_ul_('PPE1',) >= 0
 b_pw_0_'C1'_'P1'_'CPU1': - 9.9 sa_0 + 10.1 sa_1 + 10.1 sa_2 + 10.1 sa_3
   + 10.1 sa_4 + gv_b_pw_(0,_'C1',_'P1',_'CPU1') >= 0
 b_pw_1_'C1'_'P1'_'CPU1': 20 sa_1 + 0.2 sa_2 - 19.8 sa_3 + 20 sa_4
   + gv_b_pw_(1,_'C1',_'P1',_'CPU1') >= 0
 b_pw_2_'C1'_'P1'_'CPU1': 20 sa_1 - 19.6 sa_2 + 20 sa_3 + 20 sa_4
   + gv_b_pw_(2,_'C1',_'P1',_'CPU1') >= 0
 b_ps_1_0_'C1'_'P1'_'CPU1': 10 sa_1 + gv_b_ps_(1,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_1_1_'C1'_'P1'_'CPU1': - 19.7 sa_1 + 10 sa_2
   + gv_b_ps_(1,_1,_'C1',_'P1',_'CPU1') >= 0
 b_ps_1_2_'C1'_'P1'_'CPU1': - 69.2 sa_1 + 10 sa_2 + 0.1 sa_3 + 0.1 sa_4
   + gv_b_ps_(1,_2,_'C1',_'P1',_'CPU1') >= 0
 b_ps_2_0_'C1'_'P1'_'CPU1': 10 sa_1 + gv_b_ps_(2,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_2_1_'C1'_'P1'_'CPU1': 10 sa_1 + gv_b_ps_(2,_1,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_2_2_'C1'_'P1'_'CPU1': 10 sa_1 + gv_b_ps_(2,_2,_'C1',_'P1',_'CPU1')
   >= 0
Bounds
End
