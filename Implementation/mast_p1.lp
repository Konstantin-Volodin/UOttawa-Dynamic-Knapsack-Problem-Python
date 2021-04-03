\ Model MasterKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  gv_b_0 + gv_b_ue_(1,_'PPE_1') + gv_b_uu_(1,_'PPE_1')
   + gv_b_uv__(1,_'PPE_1') + gv_b_ue_(2,_'PPE_1') + gv_b_uu_(2,_'PPE_1')
   + gv_b_uv__(2,_'PPE_1') + gv_b_ue_(3,_'PPE_1') + gv_b_uu_(3,_'PPE_1')
   + gv_b_uv__(3,_'PPE_1') + gv_b_ue_(4,_'PPE_1') + gv_b_uu_(4,_'PPE_1')
   + gv_b_uv__(4,_'PPE_1') + gv_b_ue_(5,_'PPE_1') + gv_b_uu_(5,_'PPE_1')
   + gv_b_uv__(5,_'PPE_1') + gv_b_ue_(6,_'PPE_1') + gv_b_uu_(6,_'PPE_1')
   + gv_b_uv__(6,_'PPE_1') + gv_b_ue_(7,_'PPE_1') + gv_b_uu_(7,_'PPE_1')
   + gv_b_uv__(7,_'PPE_1') + gv_b_ue_(8,_'PPE_1') + gv_b_uu_(8,_'PPE_1')
   + gv_b_uv__(8,_'PPE_1') + gv_b_ue_(9,_'PPE_1') + gv_b_uu_(9,_'PPE_1')
   + gv_b_uv__(9,_'PPE_1') + gv_b_ue_(10,_'PPE_1') + gv_b_uu_(10,_'PPE_1')
   + gv_b_uv__(10,_'PPE_1') + gv_b_pe_('Complexity_1',_'CPU_1')
   + gv_b_pw_(1,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(2,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(3,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(4,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(5,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(6,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(7,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(8,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(9,_'Complexity_1',_'CPU_1')
   + gv_b_pw_(10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(1,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(2,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(3,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(4,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(5,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(6,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(7,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(8,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(9,_10,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_1,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_2,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_3,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_4,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_5,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_6,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_7,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_8,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_9,_'Complexity_1',_'CPU_1')
   + gv_b_ps_(10,_10,_'Complexity_1',_'CPU_1')
Subject To
 b_0: 0.05 sa_0 + gv_b_0 = 1
 b_ue_1_PPE-1: - 9 sa_0 + gv_b_ue_(1,_'PPE_1') >= 10
 b_ue_2_PPE-1: 0.5 sa_0 + gv_b_ue_(2,_'PPE_1') >= 10
 b_ue_3_PPE-1: 0.5 sa_0 + gv_b_ue_(3,_'PPE_1') >= 10
 b_ue_4_PPE-1: 0.5 sa_0 + gv_b_ue_(4,_'PPE_1') >= 10
 b_ue_5_PPE-1: 0.5 sa_0 + gv_b_ue_(5,_'PPE_1') >= 10
 b_ue_6_PPE-1: 0.5 sa_0 + gv_b_ue_(6,_'PPE_1') >= 10
 b_ue_7_PPE-1: 0.5 sa_0 + gv_b_ue_(7,_'PPE_1') >= 10
 b_ue_8_PPE-1: 0.5 sa_0 + gv_b_ue_(8,_'PPE_1') >= 10
 b_ue_9_PPE-1: 0.5 sa_0 + gv_b_ue_(9,_'PPE_1') >= 10
 b_ue_10_PPE-1: 0.5 sa_0 + gv_b_ue_(10,_'PPE_1') >= 10
 b_uu_1_PPE-1: gv_b_uu_(1,_'PPE_1') >= 0
 b_uu_2_PPE-1: gv_b_uu_(2,_'PPE_1') >= 0
 b_uu_3_PPE-1: gv_b_uu_(3,_'PPE_1') >= 0
 b_uu_4_PPE-1: gv_b_uu_(4,_'PPE_1') >= 0
 b_uu_5_PPE-1: gv_b_uu_(5,_'PPE_1') >= 0
 b_uu_6_PPE-1: gv_b_uu_(6,_'PPE_1') >= 0
 b_uu_7_PPE-1: gv_b_uu_(7,_'PPE_1') >= 0
 b_uu_8_PPE-1: gv_b_uu_(8,_'PPE_1') >= 0
 b_uu_9_PPE-1: gv_b_uu_(9,_'PPE_1') >= 0
 b_uu_10_PPE-1: gv_b_uu_(10,_'PPE_1') >= 0
 b_uv_1_PPE-1: gv_b_uv__(1,_'PPE_1') >= 0
 b_uv_2_PPE-1: gv_b_uv__(2,_'PPE_1') >= 0
 b_uv_3_PPE-1: gv_b_uv__(3,_'PPE_1') >= 0
 b_uv_4_PPE-1: gv_b_uv__(4,_'PPE_1') >= 0
 b_uv_5_PPE-1: gv_b_uv__(5,_'PPE_1') >= 0
 b_uv_6_PPE-1: gv_b_uv__(6,_'PPE_1') >= 0
 b_uv_7_PPE-1: gv_b_uv__(7,_'PPE_1') >= 0
 b_uv_8_PPE-1: gv_b_uv__(8,_'PPE_1') >= 0
 b_uv_9_PPE-1: gv_b_uv__(9,_'PPE_1') >= 0
 b_uv_10_PPE-1: gv_b_uv__(10,_'PPE_1') >= 0
 b_pe_Complexity-1_CPU-1: - 0.95 sa_0 + gv_b_pe_('Complexity_1',_'CPU_1')
   >= 1
 b_pw_1_Complexity-1_CPU-1: gv_b_pw_(1,_'Complexity_1',_'CPU_1') >= 0
 b_pw_2_Complexity-1_CPU-1: gv_b_pw_(2,_'Complexity_1',_'CPU_1') >= 0
 b_pw_3_Complexity-1_CPU-1: gv_b_pw_(3,_'Complexity_1',_'CPU_1') >= 0
 b_pw_4_Complexity-1_CPU-1: gv_b_pw_(4,_'Complexity_1',_'CPU_1') >= 0
 b_pw_5_Complexity-1_CPU-1: gv_b_pw_(5,_'Complexity_1',_'CPU_1') >= 0
 b_pw_6_Complexity-1_CPU-1: gv_b_pw_(6,_'Complexity_1',_'CPU_1') >= 0
 b_pw_7_Complexity-1_CPU-1: gv_b_pw_(7,_'Complexity_1',_'CPU_1') >= 0
 b_pw_8_Complexity-1_CPU-1: gv_b_pw_(8,_'Complexity_1',_'CPU_1') >= 0
 b_pw_9_Complexity-1_CPU-1: gv_b_pw_(9,_'Complexity_1',_'CPU_1') >= 0
 b_pw_10_Complexity-1_CPU-1: gv_b_pw_(10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(1,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(1,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(2,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(2,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(3,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(3,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(4,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(4,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(5,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(5,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(6,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(6,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(7,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(7,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(8,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(8,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(9,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(9,_10,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_1,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_1,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_2,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_2,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_3,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_3,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_4,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_4,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_5,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_5,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_6,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_6,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_7,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_7,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_8,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_8,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_9,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_9,_'Complexity_1',_'CPU_1') >= 0
 b_ps_(10,_10,_'Complexity_1',_'CPU_1'):
   gv_b_ps_(10,_10,_'Complexity_1',_'CPU_1') >= 0
Bounds
End
