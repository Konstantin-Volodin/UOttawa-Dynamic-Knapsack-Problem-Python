\ Model MasterKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  15177.8743666484 sa_1 + 15157.1244269676 sa_2 + 15157.1244269676 sa_3
   + 15167.499396808 sa_4
Subject To
 b_0: 0.05 sa_0 + 0.05 sa_1 + 0.05 sa_2 + 0.05 sa_3 + 0.05 sa_4 = 1
 b_ue_1_PPE-1: - 9 sa_0 + 19.75 sa_1 + 19.75 sa_2 + 19.75 sa_3 + 19.75 sa_4
   >= 10
 b_ue_2_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_3_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_4_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_5_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_6_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_7_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_8_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_9_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_ue_10_PPE-1: 0.5 sa_0 + 5.5 sa_1 + 5.5 sa_2 + 5.5 sa_3 + 5.5 sa_4 >= 10
 b_uu_1_PPE-1: 15.75 sa_1 - 58.5 sa_2 + 1.5 sa_3 + 1.5 sa_4 >= 0
 b_uu_2_PPE-1: - 14.25 sa_1 + 1.5 sa_2 + 1.5 sa_3 + 1.5 sa_4 >= 0
 b_uu_3_PPE-1: - 20.9 sa_1 + 1.5 sa_2 + 1.5 sa_3 + 30 sa_4 >= 0
 b_uu_4_PPE-1: - 14.5 sa_1 + 30 sa_2 + 30 sa_3 >= 0
 b_uu_5_PPE-1: 1.5 sa_1 - 36.5 sa_4 >= 0
 b_uu_6_PPE-1: 1.5 sa_1 - 4 sa_2 - 32.5 sa_3 + 30 sa_4 >= 0
 b_uu_7_PPE-1: 7.2 sa_1 - 32.5 sa_2 + 30 sa_3 - 28.5 sa_4 >= 0
 b_uu_8_PPE-1: - 10.5 sa_1 + 30 sa_2 + 30 sa_4 >= 0
 b_uu_9_PPE-1: 1.5 sa_1 - 4 sa_4 >= 0
 b_uu_10_PPE-1: 30 sa_1 >= 0
 b_uv_1_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_2_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_3_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_4_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_5_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_6_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_7_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_8_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_9_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_uv_10_PPE-1: 15 sa_1 + 15 sa_2 + 15 sa_3 + 15 sa_4 >= 0
 b_pe_Complexity-1_CPU-1: - 0.95 sa_0 + 3.05 sa_1 + 3.05 sa_2 + 3.05 sa_3
   + 3.05 sa_4 >= 1
 b_pw_1_Complexity-1_CPU-1: 4 sa_1 + 0.2 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_2_Complexity-1_CPU-1: 4 sa_1 + 2.1 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_3_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_4_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_5_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_6_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_7_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 0.2 sa_4 >= 0
 b_pw_8_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 0.2 sa_3 + 4 sa_4 >= 0
 b_pw_9_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4 >= 0
 b_pw_10_Complexity-1_CPU-1: 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4 >= 0
 b_ps_(1,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(1,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_4,_'Complexity_1',_'CPU_1'): 1.15 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_5,_'Complexity_1',_'CPU_1'): 4 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_6,_'Complexity_1',_'CPU_1'): 4 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_7,_'Complexity_1',_'CPU_1'): 4 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_8,_'Complexity_1',_'CPU_1'): 2.1 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(1,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 0.2 sa_2 - 3.6 sa_3
   - 3.6 sa_4 >= 0
 b_ps_(2,_1,_'Complexity_1',_'CPU_1'): 7.8 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(2,_2,_'Complexity_1',_'CPU_1'): 4 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_3,_'Complexity_1',_'CPU_1'): 4 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_4,_'Complexity_1',_'CPU_1'): 3.05 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 4 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(2,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 4 sa_2 + 0.2 sa_3
   - 3.6 sa_4 >= 0
 b_ps_(3,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(3,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 - 3.8 sa_3
   + 4 sa_4 >= 0
 b_ps_(3,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 4 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(3,_10,_'Complexity_1',_'CPU_1'): 4 sa_1 - 7.6 sa_2 - 7.6 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(4,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(4,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(4,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 - 3.8 sa_3
   + 4 sa_4 >= 0
 b_ps_(4,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(4,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 0.2 sa_2 - 3.6 sa_3
   + 4 sa_4 >= 0
 b_ps_(5,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(5,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(5,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 4 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(5,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 4 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(6,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(6,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 - 3.8 sa_3
   + 4 sa_4 >= 0
 b_ps_(6,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 4 sa_3 >= 0
 b_ps_(6,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 4 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(7,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(7,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(7,_8,_'Complexity_1',_'CPU_1'): 2.1 sa_1 + 0.2 sa_2 - 3.8 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(7,_9,_'Complexity_1',_'CPU_1'): 4 sa_1 - 3.8 sa_2 - 3.8 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(7,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 0.2 sa_2 - 3.6 sa_3
   - 3.6 sa_4 >= 0
 b_ps_(8,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(8,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(8,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 4 sa_3
   + 4 sa_4 >= 0
 b_ps_(8,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 0.2 sa_2 - 3.6 sa_3
   + 4 sa_4 >= 0
 b_ps_(9,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(9,_2,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_3,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_4,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_5,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_6,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_7,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 + 0.2 sa_3
   + 0.2 sa_4 >= 0
 b_ps_(9,_8,_'Complexity_1',_'CPU_1'): 0.2 sa_1 + 0.2 sa_2 - 3.8 sa_3
   + 7.8 sa_4 >= 0
 b_ps_(9,_9,_'Complexity_1',_'CPU_1'): 0.2 sa_1 - 3.8 sa_2 + 7.8 sa_3
   - 3.8 sa_4 >= 0
 b_ps_(9,_10,_'Complexity_1',_'CPU_1'): - 3.6 sa_1 + 4 sa_2 + 7.8 sa_3
   - 7.6 sa_4 >= 0
 b_ps_(10,_1,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_2,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_3,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_4,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_5,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_6,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 + 4 sa_4
   >= 0
 b_ps_(10,_7,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_3 >= 0
 b_ps_(10,_8,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_2 + 4 sa_4 >= 0
 b_ps_(10,_9,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_4 >= 0
 b_ps_(10,_10,_'Complexity_1',_'CPU_1'): 4 sa_1 + 4 sa_4 >= 0
Bounds
End
