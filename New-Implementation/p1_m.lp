\ Model MasterKnapsack_copy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  gv_b_0 + gv_b_ul_('PPE1',) + gv_b_pw_(0,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(0,_'C2',_'P1',_'CPU1') + gv_b_pw_(1,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(1,_'C2',_'P1',_'CPU1') + gv_b_pw_(2,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(2,_'C2',_'P1',_'CPU1') + gv_b_pw_(3,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(3,_'C2',_'P1',_'CPU1') + gv_b_pw_(4,_'C1',_'P1',_'CPU1')
   + gv_b_pw_(4,_'C2',_'P1',_'CPU1') + gv_b_ps_(1,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_0,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(1,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_1,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(1,_2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_2,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(1,_3,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_3,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(1,_4,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(1,_4,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(2,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_0,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(2,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_1,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(2,_2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_2,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(2,_3,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_3,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(2,_4,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(2,_4,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(3,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(3,_0,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(3,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(3,_1,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(3,_2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(3,_2,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(3,_3,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(3,_3,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(3,_4,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(3,_4,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(4,_0,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(4,_0,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(4,_1,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(4,_1,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(4,_2,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(4,_2,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(4,_3,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(4,_3,_'C2',_'P1',_'CPU1')
   + gv_b_ps_(4,_4,_'C1',_'P1',_'CPU1')
   + gv_b_ps_(4,_4,_'C2',_'P1',_'CPU1')
Subject To
 b_0: 0.01 sa_0 + 0.01 sa_1 + 0.01 sa_2 + 0.01 sa_3 + 0.01 sa_4 + 0.01 sa_5
   + gv_b_0 = 1
 b_ul_'PPE1': 150 sa_2 + gv_b_ul_('PPE1',) >= 0
 b_pw_0_'C1'_'P1'_'CPU1': - 8.91 sa_0 + 127.09 sa_1 + 127.09 sa_2
   + 127.09 sa_3 + 127.09 sa_4 + 127.09 sa_5
   + gv_b_pw_(0,_'C1',_'P1',_'CPU1') >= 9
 b_pw_0_'C2'_'P1'_'CPU1': sa_1 + sa_2 + sa_3
   + gv_b_pw_(0,_'C2',_'P1',_'CPU1') >= 0
 b_pw_1_'C1'_'P1'_'CPU1': 136 sa_1 + 1.36 sa_2 - 134.64 sa_3 + 136 sa_4
   + 136 sa_5 + gv_b_pw_(1,_'C1',_'P1',_'CPU1') >= 8
 b_pw_1_'C2'_'P1'_'CPU1': 0.01 sa_1 + 0.01 sa_2 - 0.99 sa_3 + sa_4 + sa_5
   + gv_b_pw_(1,_'C2',_'P1',_'CPU1') >= 0
 b_pw_2_'C1'_'P1'_'CPU1': 136 sa_1 - 134.64 sa_2 + 136 sa_3 - 134.64 sa_4
   - 134.64 sa_5 + gv_b_pw_(2,_'C1',_'P1',_'CPU1') >= 8
 b_pw_2_'C2'_'P1'_'CPU1': 0.01 sa_1 - 0.99 sa_2 + sa_3 - 0.99 sa_4
   - 0.99 sa_5 + gv_b_pw_(2,_'C2',_'P1',_'CPU1') >= 0
 b_pw_3_'C1'_'P1'_'CPU1': 90.559 sa_1 + 136 sa_2 + 136 sa_3
   + gv_b_pw_(3,_'C1',_'P1',_'CPU1') >= 7
 b_pw_3_'C2'_'P1'_'CPU1': - 5.039 sa_1 + sa_2 + sa_3 + sa_4 + sa_5
   + gv_b_pw_(3,_'C2',_'P1',_'CPU1') >= 4
 b_pw_4_'C1'_'P1'_'CPU1': 136 sa_1 - 121.176 sa_2
   + gv_b_pw_(4,_'C1',_'P1',_'CPU1') >= 7
 b_pw_4_'C2'_'P1'_'CPU1': 0.01 sa_1 - 14.444 sa_2 + sa_3 + sa_4 + sa_5
   + gv_b_pw_(4,_'C2',_'P1',_'CPU1') >= 4
 b_ps_1_0_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(1,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_1_0_'C2'_'P1'_'CPU1': sa_1 + gv_b_ps_(1,_0,_'C2',_'P1',_'CPU1') >= 0
 b_ps_1_1_'C1'_'P1'_'CPU1': 136 sa_1 - 9.9 sa_4 - 9.9 sa_5
   + gv_b_ps_(1,_1,_'C1',_'P1',_'CPU1') >= 1
 b_ps_1_1_'C2'_'P1'_'CPU1': - 0.98 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(1,_1,_'C2',_'P1',_'CPU1') >= 0
 b_ps_1_2_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(1,_2,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_1_2_'C2'_'P1'_'CPU1': - 0.98 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(1,_2,_'C2',_'P1',_'CPU1') >= 0
 b_ps_1_3_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(1,_3,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_1_3_'C2'_'P1'_'CPU1': - 0.98 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(1,_3,_'C2',_'P1',_'CPU1') >= 0
 b_ps_1_4_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(1,_4,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_1_4_'C2'_'P1'_'CPU1': - 3.95 sa_1 + 0.01 sa_2 + sa_3 + sa_4 + sa_5
   + gv_b_ps_(1,_4,_'C2',_'P1',_'CPU1') >= 0
 b_ps_2_0_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(2,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_2_0_'C2'_'P1'_'CPU1': sa_1 + gv_b_ps_(2,_0,_'C2',_'P1',_'CPU1') >= 0
 b_ps_2_1_'C1'_'P1'_'CPU1': 133.03 sa_1 - 6.93 sa_4 - 124.74 sa_5
   + gv_b_ps_(2,_1,_'C1',_'P1',_'CPU1') >= 1
 b_ps_2_1_'C2'_'P1'_'CPU1': 0.01 sa_1 + gv_b_ps_(2,_1,_'C2',_'P1',_'CPU1')
   >= 0
 b_ps_2_2_'C1'_'P1'_'CPU1': 136 sa_1 - 128.7 sa_5
   + gv_b_ps_(2,_2,_'C1',_'P1',_'CPU1') >= 1
 b_ps_2_2_'C2'_'P1'_'CPU1': 0.01 sa_1 - 1.98 sa_2 + sa_3 - 0.99 sa_4
   + gv_b_ps_(2,_2,_'C2',_'P1',_'CPU1') >= 0
 b_ps_2_3_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(2,_3,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_2_3_'C2'_'P1'_'CPU1': 0.01 sa_1 - 1.98 sa_2 + sa_3 - 0.99 sa_4
   + gv_b_ps_(2,_3,_'C2',_'P1',_'CPU1') >= 0
 b_ps_2_4_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(2,_4,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_2_4_'C2'_'P1'_'CPU1': - 0.98 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(2,_4,_'C2',_'P1',_'CPU1') >= 0
 b_ps_3_0_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(3,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_3_0_'C2'_'P1'_'CPU1': sa_1 + gv_b_ps_(3,_0,_'C2',_'P1',_'CPU1') >= 0
 b_ps_3_1_'C1'_'P1'_'CPU1': 136 sa_1 - 117.81 sa_4 + 136 sa_5
   + gv_b_ps_(3,_1,_'C1',_'P1',_'CPU1') >= 1
 b_ps_3_1_'C2'_'P1'_'CPU1': 0.01 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(3,_1,_'C2',_'P1',_'CPU1') >= 1
 b_ps_3_2_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(3,_2,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_3_2_'C2'_'P1'_'CPU1': 0.01 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(3,_2,_'C2',_'P1',_'CPU1') >= 1
 b_ps_3_3_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(3,_3,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_3_3_'C2'_'P1'_'CPU1': 0.01 sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(3,_3,_'C2',_'P1',_'CPU1') >= 1
 b_ps_3_4_'C1'_'P1'_'CPU1': 133.327 sa_1
   + gv_b_ps_(3,_4,_'C1',_'P1',_'CPU1') >= 1
 b_ps_3_4_'C2'_'P1'_'CPU1': - 1.277 sa_1 + 0.01 sa_2 + sa_3 + sa_4
   + gv_b_ps_(3,_4,_'C2',_'P1',_'CPU1') >= 1
 b_ps_4_0_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(4,_0,_'C1',_'P1',_'CPU1')
   >= 0
 b_ps_4_0_'C2'_'P1'_'CPU1': sa_1 + gv_b_ps_(4,_0,_'C2',_'P1',_'CPU1') >= 0
 b_ps_4_1_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(4,_1,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_4_1_'C2'_'P1'_'CPU1': sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(4,_1,_'C2',_'P1',_'CPU1') >= 1
 b_ps_4_2_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(4,_2,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_4_2_'C2'_'P1'_'CPU1': sa_1 + sa_2 + sa_3
   + gv_b_ps_(4,_2,_'C2',_'P1',_'CPU1') >= 1
 b_ps_4_3_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(4,_3,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_4_3_'C2'_'P1'_'CPU1': sa_1 + sa_2 + sa_3
   + gv_b_ps_(4,_3,_'C2',_'P1',_'CPU1') >= 1
 b_ps_4_4_'C1'_'P1'_'CPU1': 136 sa_1 + gv_b_ps_(4,_4,_'C1',_'P1',_'CPU1')
   >= 1
 b_ps_4_4_'C2'_'P1'_'CPU1': sa_1 + sa_2 + sa_3 + sa_4
   + gv_b_ps_(4,_4,_'C2',_'P1',_'CPU1') >= 1
Bounds
End
