from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Data Classes
@dataclass(frozen=True)
class state:
    ue_tp: Dict[ Tuple[str],float ] # units expected on time t, ppe p
    uu_tp: Dict[ Tuple[str],float ] # units used on time t, ppe p
    uv_tp: Dict[ Tuple[str],float ] # units violated on time t, ppe p
    pe_dc: Dict[ Tuple[str],float ] # new patient arrivals of complexity d, cpu c
    pw_mdc: Dict[ Tuple[str],float ] # patients waiting for m periods, of complexity d, cpu c
    ps_tmdc: Dict[ Tuple[str],float ] # patients scheduled into time t, who have waited for m periods, of complexity d, cpu c
@dataclass(frozen=True)
class action:
    sc_tmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods to schedule into t (m of 0 corresponds to pe)
    rsc_ttpmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods, to reschedule from t to tp 
@dataclass(frozen=True)
class constraint_parameter:
    lhs_param: Dict[ Tuple[str],float ]
    rhs_param: Dict[ Tuple[str],float ]
    sign: Dict[ Tuple[str],str ]
    name:  Dict[ Tuple[str],str ]


def generate_sub_problem(input_data, betas):
    indices = input_data.indices
    arrival = input_data.arrival
    ppe_data = input_data.ppe_data
    model_param = input_data.model_param
    transition = input_data.transition
    usage = input_data.usage

    M = model_param.M
    gamma = model_param.gamma

    # Initializes model
    sub_model = gp.Model('SubproblemKnapsack')

    var_ue = {}
    var_uu = {}
    var_uv = {}
    var_pw = {}
    var_pe = {}
    var_ps = {}
    var_sc = {}
    var_rsc = {}

    # Decision Variables
        # States
    for tp in itertools.product(indices['t'], indices['p']):
        var_ue[tp] = sub_model.addVar(name=f's_ue_{tp}')
        var_uu[tp] = sub_model.addVar(name=f's_uu_{tp}')
        var_uv[tp] = sub_model.addVar(name=f's_uv_{tp}')
    for dc in itertools.product(indices['d'], indices['c']):
        var_pe[dc] = sub_model.addVar(name='s_pe_{dc}')
    for mdc in itertools.product(indices['m'], indices['d'], indices['c']):
        if mdc[0] == 0: continue
        var_pw[mdc] = sub_model.addVar(name='s_pw_{mdc}')
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_ps[tmdc] = sub_model.addVar(name='s_ps_{tmdc}')
        # Actions
    for tmdc in itertools.product(indices['t'], indices['m'], indices['d'], indices['c']):
        var_sc[tmdc] = sub_model.addVar(name='a_sc_{tmdc}')
    for ttpmdc in itertools.product(indices['t'], indices['t'], indices['m'], indices['d'], indices['c']):
        var_rsc[ttpmdc] = sub_model.addVar(name='a_rsc_{ttpmdc}')

    # Constraints

    # 1) PPE Capacity
    for tp in itertools.product(indices['t'], indices['p']):


    # Main Objective Function