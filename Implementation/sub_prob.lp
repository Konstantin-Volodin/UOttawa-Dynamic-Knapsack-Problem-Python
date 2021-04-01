\ Model SubmodelKnapsack
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  100 s_uv_(1,_'PPE_1') - s_ue_(2,_'PPE_1') + 100 s_uv_(2,_'PPE_1')
   - s_ue_(3,_'PPE_1') + 100 s_uv_(3,_'PPE_1') - s_ue_(4,_'PPE_1')
   + 100 s_uv_(4,_'PPE_1') - s_ue_(5,_'PPE_1') + 100 s_uv_(5,_'PPE_1')
   - s_ue_(6,_'PPE_1') + 100 s_uv_(6,_'PPE_1') - s_ue_(7,_'PPE_1')
   + 100 s_uv_(7,_'PPE_1') - s_ue_(8,_'PPE_1') + 100 s_uv_(8,_'PPE_1')
   - s_ue_(9,_'PPE_1') + 100 s_uv_(9,_'PPE_1') - s_ue_(10,_'PPE_1')
   + 100 s_uv_(10,_'PPE_1') + s_pe_('Complexity_1',_'CPU_1')
   + 1.1 s_pw_(1,_'Complexity_1',_'CPU_1')
   + 1.21 s_pw_(2,_'Complexity_1',_'CPU_1')
   + 1.331 s_pw_(3,_'Complexity_1',_'CPU_1')
   + 1.4641 s_pw_(4,_'Complexity_1',_'CPU_1')
   + 1.61051 s_pw_(5,_'Complexity_1',_'CPU_1')
   + 1.771561000000001 s_pw_(6,_'Complexity_1',_'CPU_1')
   + 1.948717100000001 s_pw_(7,_'Complexity_1',_'CPU_1')
   + 2.143588810000002 s_pw_(8,_'Complexity_1',_'CPU_1')
   + 2.357947691000002 s_pw_(9,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_pw_(10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(1,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(1,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(2,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(2,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(3,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(3,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(4,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(4,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(5,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(5,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(6,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(6,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(7,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(7,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(8,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(8,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(9,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(9,_10,_'Complexity_1',_'CPU_1')
   + 0 s_ps_(10,_0,_'Complexity_1',_'CPU_1')
   + 2.593742460100002 s_ps_(10,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_2,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_3,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_4,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_5,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(1,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(2,_1,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_3,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_4,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_5,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(2,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(3,_2,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_4,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_5,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(3,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(4,_3,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_5,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(4,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(5,_4,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(5,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_4,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(6,_5,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(6,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_4,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_5,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(7,_6,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(7,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_4,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_5,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_6,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(8,_7,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_9,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(8,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_4,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_5,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_6,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_7,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(9,_8,_10,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_1,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_2,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_3,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_4,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_5,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_6,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_7,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_8,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_9,_'Complexity_1',_'CPU_1')
   + 3 a_rsc_(9,_10,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_1,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_2,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_3,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_4,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_5,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_6,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_7,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_8,_10,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_1,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_2,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_3,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_4,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_5,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_6,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_7,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_8,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_9,_'Complexity_1',_'CPU_1')
   - 3 a_rsc_(10,_9,_10,_'Complexity_1',_'CPU_1') + 84.5 Constant
Subject To
 ppe_capacity_(1,_'PPE_1'): s_ue_(1,_'PPE_1') - s_uu_(1,_'PPE_1')
   + s_uv_(1,_'PPE_1') - a_sc_(1,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(2,_'PPE_1'): s_ue_(2,_'PPE_1') - s_uu_(2,_'PPE_1')
   + s_uv_(2,_'PPE_1') - a_sc_(2,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(3,_'PPE_1'): s_ue_(3,_'PPE_1') - s_uu_(3,_'PPE_1')
   + s_uv_(3,_'PPE_1') - a_sc_(3,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(4,_'PPE_1'): s_ue_(4,_'PPE_1') - s_uu_(4,_'PPE_1')
   + s_uv_(4,_'PPE_1') - a_sc_(4,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(5,_'PPE_1'): s_ue_(5,_'PPE_1') - s_uu_(5,_'PPE_1')
   + s_uv_(5,_'PPE_1') - a_sc_(5,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(6,_'PPE_1'): s_ue_(6,_'PPE_1') - s_uu_(6,_'PPE_1')
   + s_uv_(6,_'PPE_1') - a_sc_(6,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(7,_'PPE_1'): s_ue_(7,_'PPE_1') - s_uu_(7,_'PPE_1')
   + s_uv_(7,_'PPE_1') - a_sc_(7,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(8,_'PPE_1'): s_ue_(8,_'PPE_1') - s_uu_(8,_'PPE_1')
   + s_uv_(8,_'PPE_1') - a_sc_(8,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(9,_'PPE_1'): s_ue_(9,_'PPE_1') - s_uu_(9,_'PPE_1')
   + s_uv_(9,_'PPE_1') - a_sc_(9,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_10,_'Complexity_1',_'CPU_1') >= 0
 ppe_capacity_(10,_'PPE_1'): s_ue_(10,_'PPE_1') - s_uu_(10,_'PPE_1')
   + s_uv_(10,_'PPE_1') - a_sc_(10,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 resc_bound_(1,_1,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_1,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_1,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(1,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(1,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(2,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(2,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(3,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(3,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(4,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(4,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(5,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(5,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(6,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(6,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(7,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(7,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(8,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(8,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(9,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(9,_10,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_2,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_2,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_3,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_3,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_4,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_4,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_5,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_5,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_6,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_6,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_7,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_7,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_8,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_8,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_9,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_9,_10,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_1,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_1,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_2,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_2,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_3,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_3,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_4,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_4,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_5,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_5,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_6,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_6,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_7,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_7,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_8,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_8,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_9,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_9,_'Complexity_1',_'CPU_1') = 0
 resc_bound_(10,_10,_10,_'Complexity_1',_'CPU_1'):
   a_rsc_(10,_10,_10,_'Complexity_1',_'CPU_1') = 0
 consistent_resc_(1,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(1,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(1,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(2,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(2,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(3,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(3,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(4,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(4,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(5,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(5,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(6,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(6,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(7,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(7,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(8,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(8,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(9,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(9,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_1,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_1,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_2,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_2,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_3,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_3,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_4,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_4,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_5,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_5,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_6,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_6,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_7,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_7,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_8,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_8,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_9,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_9,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_resc_(10,_10,_'Complexity_1',_'CPU_1'):
   s_ps_(10,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_1,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_2,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_3,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_4,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_5,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_6,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_7,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_8,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_9,_10,_'Complexity_1',_'CPU_1')
   - a_rsc_(10,_10,_10,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(0,_'Complexity_1',_'CPU_1'):
   s_pe_('Complexity_1',_'CPU_1') - a_sc_(1,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_0,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_0,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(1,_'Complexity_1',_'CPU_1'):
   s_pw_(1,_'Complexity_1',_'CPU_1') - a_sc_(1,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_1,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_1,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(2,_'Complexity_1',_'CPU_1'):
   s_pw_(2,_'Complexity_1',_'CPU_1') - a_sc_(1,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_2,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_2,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(3,_'Complexity_1',_'CPU_1'):
   s_pw_(3,_'Complexity_1',_'CPU_1') - a_sc_(1,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_3,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_3,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(4,_'Complexity_1',_'CPU_1'):
   s_pw_(4,_'Complexity_1',_'CPU_1') - a_sc_(1,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_4,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_4,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(5,_'Complexity_1',_'CPU_1'):
   s_pw_(5,_'Complexity_1',_'CPU_1') - a_sc_(1,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_5,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_5,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(6,_'Complexity_1',_'CPU_1'):
   s_pw_(6,_'Complexity_1',_'CPU_1') - a_sc_(1,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_6,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_6,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(7,_'Complexity_1',_'CPU_1'):
   s_pw_(7,_'Complexity_1',_'CPU_1') - a_sc_(1,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_7,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_7,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(8,_'Complexity_1',_'CPU_1'):
   s_pw_(8,_'Complexity_1',_'CPU_1') - a_sc_(1,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_8,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_8,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(9,_'Complexity_1',_'CPU_1'):
   s_pw_(9,_'Complexity_1',_'CPU_1') - a_sc_(1,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_9,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_9,_'Complexity_1',_'CPU_1') >= 0
 consistent_sch_(10,_'Complexity_1',_'CPU_1'):
   s_pw_(10,_'Complexity_1',_'CPU_1')
   - a_sc_(1,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(2,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(3,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(4,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(5,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(6,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(7,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(8,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(9,_10,_'Complexity_1',_'CPU_1')
   - a_sc_(10,_10,_'Complexity_1',_'CPU_1') >= 0
Bounds
 s_ue_(1,_'PPE_1') <= 15
 s_uu_(1,_'PPE_1') <= 30
 s_uv_(1,_'PPE_1') <= 15
 s_ue_(2,_'PPE_1') <= 15
 s_uu_(2,_'PPE_1') <= 30
 s_uv_(2,_'PPE_1') <= 15
 s_ue_(3,_'PPE_1') <= 15
 s_uu_(3,_'PPE_1') <= 30
 s_uv_(3,_'PPE_1') <= 15
 s_ue_(4,_'PPE_1') <= 15
 s_uu_(4,_'PPE_1') <= 30
 s_uv_(4,_'PPE_1') <= 15
 s_ue_(5,_'PPE_1') <= 15
 s_uu_(5,_'PPE_1') <= 30
 s_uv_(5,_'PPE_1') <= 15
 s_ue_(6,_'PPE_1') <= 15
 s_uu_(6,_'PPE_1') <= 30
 s_uv_(6,_'PPE_1') <= 15
 s_ue_(7,_'PPE_1') <= 15
 s_uu_(7,_'PPE_1') <= 30
 s_uv_(7,_'PPE_1') <= 15
 s_ue_(8,_'PPE_1') <= 15
 s_uu_(8,_'PPE_1') <= 30
 s_uv_(8,_'PPE_1') <= 15
 s_ue_(9,_'PPE_1') <= 15
 s_uu_(9,_'PPE_1') <= 30
 s_uv_(9,_'PPE_1') <= 15
 s_ue_(10,_'PPE_1') <= 15
 s_uu_(10,_'PPE_1') <= 30
 s_uv_(10,_'PPE_1') <= 15
 s_pe_('Complexity_1',_'CPU_1') <= 4
 s_pw_(1,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(2,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(3,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(4,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(5,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(6,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(7,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(8,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(9,_'Complexity_1',_'CPU_1') <= 4
 s_pw_(10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(1,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(2,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(3,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(4,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(5,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(6,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(7,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(8,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(9,_10,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_0,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_1,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_2,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_3,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_4,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_5,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_6,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_7,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_8,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_9,_'Complexity_1',_'CPU_1') <= 4
 s_ps_(10,_10,_'Complexity_1',_'CPU_1') <= 4
 Constant = 1
End
