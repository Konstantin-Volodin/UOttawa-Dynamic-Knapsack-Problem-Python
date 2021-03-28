\ Model MasterKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  gv_beta_0 + gv_beta_ue_tp_(1,_'PPE_1') + gv_beta_uu_tp_(1,_'PPE_1')
   + gv_beta_uv_tp_(1,_'PPE_1') + gv_beta_ue_tp_(2,_'PPE_1')
   + gv_beta_uu_tp_(2,_'PPE_1') + gv_beta_uv_tp_(2,_'PPE_1')
   + gv_beta_ue_tp_(3,_'PPE_1') + gv_beta_uu_tp_(3,_'PPE_1')
   + gv_beta_uv_tp_(3,_'PPE_1') + gv_beta_ue_tp_(4,_'PPE_1')
   + gv_beta_uu_tp_(4,_'PPE_1') + gv_beta_uv_tp_(4,_'PPE_1')
   + gv_beta_ue_tp_(5,_'PPE_1') + gv_beta_uu_tp_(5,_'PPE_1')
   + gv_beta_uv_tp_(5,_'PPE_1') + gv_beta_ue_tp_(6,_'PPE_1')
   + gv_beta_uu_tp_(6,_'PPE_1') + gv_beta_uv_tp_(6,_'PPE_1')
   + gv_beta_ue_tp_(7,_'PPE_1') + gv_beta_uu_tp_(7,_'PPE_1')
   + gv_beta_uv_tp_(7,_'PPE_1') + gv_beta_ue_tp_(8,_'PPE_1')
   + gv_beta_uu_tp_(8,_'PPE_1') + gv_beta_uv_tp_(8,_'PPE_1')
   + gv_beta_ue_tp_(9,_'PPE_1') + gv_beta_uu_tp_(9,_'PPE_1')
   + gv_beta_uv_tp_(9,_'PPE_1') + gv_beta_ue_tp_(10,_'PPE_1')
   + gv_beta_uu_tp_(10,_'PPE_1') + gv_beta_uv_tp_(10,_'PPE_1')
   + gv_beta_pe_dc_('Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(1,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(2,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(3,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(4,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(5,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(6,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(7,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(8,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(9,_'Complexity_1',_'CPU_1')
   + gv_beta_pw_mdc_(10,_'Complexity_1',_'CPU_1')
Subject To
 beta_0: 0.05 sa_0 + gv_beta_0 = 1
 beta_ue_tp_1_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(1,_'PPE_1') >= 10
 beta_ue_tp_2_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(2,_'PPE_1') >= 10
 beta_ue_tp_3_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(3,_'PPE_1') >= 10
 beta_ue_tp_4_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(4,_'PPE_1') >= 10
 beta_ue_tp_5_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(5,_'PPE_1') >= 10
 beta_ue_tp_6_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(6,_'PPE_1') >= 10
 beta_ue_tp_7_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(7,_'PPE_1') >= 10
 beta_ue_tp_8_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(8,_'PPE_1') >= 10
 beta_ue_tp_9_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(9,_'PPE_1') >= 10
 beta_ue_tp_10_PPE-1: 0.5 sa_0 + gv_beta_ue_tp_(10,_'PPE_1') >= 10
 beta_uu_tp_1_PPE-1: gv_beta_uu_tp_(1,_'PPE_1') >= 0
 beta_uu_tp_2_PPE-1: gv_beta_uu_tp_(2,_'PPE_1') >= 0
 beta_uu_tp_3_PPE-1: gv_beta_uu_tp_(3,_'PPE_1') >= 0
 beta_uu_tp_4_PPE-1: gv_beta_uu_tp_(4,_'PPE_1') >= 0
 beta_uu_tp_5_PPE-1: gv_beta_uu_tp_(5,_'PPE_1') >= 0
 beta_uu_tp_6_PPE-1: gv_beta_uu_tp_(6,_'PPE_1') >= 0
 beta_uu_tp_7_PPE-1: gv_beta_uu_tp_(7,_'PPE_1') >= 0
 beta_uu_tp_8_PPE-1: gv_beta_uu_tp_(8,_'PPE_1') >= 0
 beta_uu_tp_9_PPE-1: gv_beta_uu_tp_(9,_'PPE_1') >= 0
 beta_uu_tp_10_PPE-1: gv_beta_uu_tp_(10,_'PPE_1') >= 0
 beta_uv_tp_1_PPE-1: gv_beta_uv_tp_(1,_'PPE_1') >= 0
 beta_uv_tp_2_PPE-1: gv_beta_uv_tp_(2,_'PPE_1') >= 0
 beta_uv_tp_3_PPE-1: gv_beta_uv_tp_(3,_'PPE_1') >= 0
 beta_uv_tp_4_PPE-1: gv_beta_uv_tp_(4,_'PPE_1') >= 0
 beta_uv_tp_5_PPE-1: gv_beta_uv_tp_(5,_'PPE_1') >= 0
 beta_uv_tp_6_PPE-1: gv_beta_uv_tp_(6,_'PPE_1') >= 0
 beta_uv_tp_7_PPE-1: gv_beta_uv_tp_(7,_'PPE_1') >= 0
 beta_uv_tp_8_PPE-1: gv_beta_uv_tp_(8,_'PPE_1') >= 0
 beta_uv_tp_9_PPE-1: gv_beta_uv_tp_(9,_'PPE_1') >= 0
 beta_uv_tp_10_PPE-1: gv_beta_uv_tp_(10,_'PPE_1') >= 0
 beta_pe_dc_Complexity-1_CPU-1: - 0.95 sa_0
   + gv_beta_pe_dc_('Complexity_1',_'CPU_1') >= 1
 beta_pw_mdc_1_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(1,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_2_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(2,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_3_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(3,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_4_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(4,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_5_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(5,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_6_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(6,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_7_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(7,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_8_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(8,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_9_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(9,_'Complexity_1',_'CPU_1') >= 0
 beta_pw_mdc_10_Complexity-1_CPU-1:
   gv_beta_pw_mdc_(10,_'Complexity_1',_'CPU_1') >= 0
Bounds
End
