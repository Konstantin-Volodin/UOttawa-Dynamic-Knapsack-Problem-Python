\ Model MasterKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  sa_0
Subject To
 beta_0: 0.05 sa_0 = 1
 beta_ue_tp_1_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_2_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_3_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_4_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_5_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_6_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_7_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_8_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_9_PPE-1: 0.5 sa_0 >= 10
 beta_ue_tp_10_PPE-1: 0.5 sa_0 >= 10
 beta_uu_tp_1_PPE-1: >= 0
 beta_uu_tp_2_PPE-1: >= 0
 beta_uu_tp_3_PPE-1: >= 0
 beta_uu_tp_4_PPE-1: >= 0
 beta_uu_tp_5_PPE-1: >= 0
 beta_uu_tp_6_PPE-1: >= 0
 beta_uu_tp_7_PPE-1: >= 0
 beta_uu_tp_8_PPE-1: >= 0
 beta_uu_tp_9_PPE-1: >= 0
 beta_uu_tp_10_PPE-1: >= 0
 beta_uv_tp_1_PPE-1: >= 0
 beta_uv_tp_2_PPE-1: >= 0
 beta_uv_tp_3_PPE-1: >= 0
 beta_uv_tp_4_PPE-1: >= 0
 beta_uv_tp_5_PPE-1: >= 0
 beta_uv_tp_6_PPE-1: >= 0
 beta_uv_tp_7_PPE-1: >= 0
 beta_uv_tp_8_PPE-1: >= 0
 beta_uv_tp_9_PPE-1: >= 0
 beta_uv_tp_10_PPE-1: >= 0
 beta_pe_dc_Complexity-1_CPU-1: 0.05 sa_0 >= 1
 beta_pw_mdc_1_Complexity-1_CPU-1: - 0.95 sa_0 >= 0
 beta_pw_mdc_2_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_3_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_4_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_5_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_6_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_7_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_8_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_9_Complexity-1_CPU-1: >= 0
 beta_pw_mdc_10_Complexity-1_CPU-1: >= 0
Bounds
End
