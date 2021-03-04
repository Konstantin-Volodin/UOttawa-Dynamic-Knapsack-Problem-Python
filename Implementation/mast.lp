\ Model MasterKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  13.923 sa_0
Subject To
 beta_0: 0.05 sa_0 = 1
 beta_ue_tp_1_PPE-1: sa_0 >= 20
 beta_ue_tp_1_PPE-2: sa_0 >= 20
 beta_ue_tp_2_PPE-1: sa_0 >= 20
 beta_ue_tp_2_PPE-2: sa_0 >= 20
 beta_ue_tp_3_PPE-1: sa_0 >= 20
 beta_ue_tp_3_PPE-2: sa_0 >= 20
 beta_ue_tp_4_PPE-1: sa_0 >= 20
 beta_ue_tp_4_PPE-2: sa_0 >= 20
 beta_ue_tp_5_PPE-1: sa_0 >= 20
 beta_ue_tp_5_PPE-2: sa_0 >= 20
 beta_ue_tp_6_PPE-1: sa_0 >= 20
 beta_ue_tp_6_PPE-2: sa_0 >= 20
 beta_ue_tp_7_PPE-1: sa_0 >= 20
 beta_ue_tp_7_PPE-2: sa_0 >= 20
 beta_uu_tp_1_PPE-1: 5.75 sa_0 >= 20
 beta_uu_tp_1_PPE-2: >= 0
 beta_uu_tp_2_PPE-1: 5.5 sa_0 >= 15
 beta_uu_tp_2_PPE-2: >= 0
 beta_uu_tp_3_PPE-1: 5.25 sa_0 >= 10
 beta_uu_tp_3_PPE-2: >= 0
 beta_uu_tp_4_PPE-1: 5 sa_0 >= 5
 beta_uu_tp_4_PPE-2: >= 0
 beta_uu_tp_5_PPE-1: >= 0
 beta_uu_tp_5_PPE-2: >= 0
 beta_uu_tp_6_PPE-1: >= 0
 beta_uu_tp_6_PPE-2: >= 0
 beta_uu_tp_7_PPE-1: >= 0
 beta_uu_tp_7_PPE-2: >= 0
 beta_pe_dc_Complexity-1_CPU-1: 0.05 sa_0 >= 1
 beta_pe_dc_Complexity-1_CPU-2: - 0.45 sa_0 >= 0.5
 beta_pe_dc_Complexity-2_CPU-1: 0.525 sa_0 >= 1
 beta_pe_dc_Complexity-2_CPU-2: 0.025 sa_0 >= 0.5
 beta_pw_mdc_1_Complexity-1_CPU-1: 0.05 sa_0 >= 1
 beta_pw_mdc_1_Complexity-2_CPU-1: - 0.45 sa_0 >= 0.5
 beta_pw_mdc_1_Complexity-1_CPU-2: 0.525 sa_0 >= 1
 beta_pw_mdc_1_Complexity-2_CPU-2: 0.025 sa_0 >= 0.5
 beta_pw_mdc_2_Complexity-1_CPU-1: 0.05 sa_0 >= 1
 beta_pw_mdc_2_Complexity-2_CPU-1: 0.025 sa_0 >= 0.5
 beta_pw_mdc_2_Complexity-1_CPU-2: 0.05 sa_0 >= 1
 beta_pw_mdc_2_Complexity-2_CPU-2: 0.025 sa_0 >= 0.5
 beta_pw_mdc_3_Complexity-1_CPU-1: 0.05 sa_0 >= 1
 beta_pw_mdc_3_Complexity-2_CPU-1: 0.025 sa_0 >= 0.5
 beta_pw_mdc_3_Complexity-1_CPU-2: 0.05 sa_0 >= 1
 beta_pw_mdc_3_Complexity-2_CPU-2: 0.025 sa_0 >= 0.5
 beta_pw_mdc_4_Complexity-1_CPU-1: - 0.95 sa_0 >= 0
 beta_pw_mdc_4_Complexity-2_CPU-1: - 0.475 sa_0 >= 0
 beta_pw_mdc_4_Complexity-1_CPU-2: - 0.95 sa_0 >= 0
 beta_pw_mdc_4_Complexity-2_CPU-2: - 0.475 sa_0 >= 0
 beta_pw_mdc_5_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_5_Complexity-2_CPU-1: >= 0
 beta_pw_mdc_5_Complexity-1_CPU-2: >= 0
 beta_pw_mdc_5_Complexity-2_CPU-2: >= 0
 beta_pw_mdc_6_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_6_Complexity-2_CPU-1: >= 0
 beta_pw_mdc_6_Complexity-1_CPU-2: >= 0
 beta_pw_mdc_6_Complexity-2_CPU-2: >= 0
 beta_pw_mdc_7_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_7_Complexity-2_CPU-1: >= 0
 beta_pw_mdc_7_Complexity-1_CPU-2: >= 0
 beta_pw_mdc_7_Complexity-2_CPU-2: >= 0
Bounds
End
