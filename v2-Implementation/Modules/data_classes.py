from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
import gurobipy as gp

# Data Import Classes
@dataclass(frozen=True)
class ppe_data_class:
    ppe_type: str
    expected_units: int
    deviation: List[int]

@dataclass(frozen=False)
class model_param_class:
    cw: float
    cc: float
    cs: List[float]
    M: float
    gamma: float
@dataclass(frozen=False)
class transition_data_class:
    wait_limit: int
    transition_rate: float
@dataclass()
class input_data_class:
    indices: Dict[ str, List[int] ]
    ppe_data: Dict[ str,ppe_data_class ]
    usage: Dict[ Tuple[str], List[int] ]
    arrival: Dict[ Tuple[str], List[int] ]
    transition: Dict[ Tuple[str], transition_data_class ]
    model_param: model_param_class
    expected_state_values: Dict[ str, Dict[Tuple[str], float] ]

# Data Export Classes

# State Action 
@dataclass(frozen=True)
class state:
    ul_p: Dict[ Tuple[str],float ] # units left over on time 1, ppe p
    pw_mdc: Dict[ Tuple[str],float ] # patients waiting for m periods, of complexity d, cpu c
    ps_tmdc: Dict[ Tuple[str],float ] # patients scheduled into time t, who have waited for m periods, of complexity d, cpu c
@dataclass(frozen=True)
class action:
    sc_tmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods to schedule into t (m of 0 corresponds to pe)
    rsc_ttpmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods, to reschedule from t to tp 
    uv_tp: Dict[ Tuple[str],float ] # unit violations on time t, ppe p

    uvb_tp: Dict[ Tuple[str],float ] # unit violations binary on time t, ppe p
    ul_p_p: Dict[ Tuple[str],float ] # unit left over - post action - time 1, ppe p
    ulb_p: Dict[ Tuple[str],float ] # unit left over binary - time 1, ppe p
    uu_p_tp: Dict[ Tuple[str],float ] # units used - post action; time t, ppe p
    pw_p_mdc: Dict[ Tuple[str],float ] # patients waiting - post action; m periods, complexity d, cpu c
    ps_p_tmdc: Dict[ Tuple[str],float ] # patients scheduled - post action; time t, m periods, complexity d, cpu c

# Gurobi Relevant Objects
@dataclass(frozen=True)
class constraint_parameter:
    lhs_param: Dict[ Tuple[str],float ]
    rhs_param: Dict[ Tuple[str],float ]
    sign: Dict[ Tuple[str],str ]
    name:  Dict[ Tuple[str],str ]
@dataclass(frozen=True)
class variables:
    s_ul: Dict[ Tuple[str],gp.Var ]
    s_pw: Dict[ Tuple[str],gp.Var ]
    s_ps: Dict[ Tuple[str],gp.Var ]

    a_sc: Dict[ Tuple[str],gp.Var ]
    a_rsc: Dict[ Tuple[str],gp.Var ]
    a_uv: Dict[ Tuple[str],gp.Var ]

    a_uvb: Dict[ Tuple[str],gp.Var ]
    a_ul_p: Dict[ Tuple[str],gp.Var ]
    a_ulb: Dict[ Tuple[str],gp.Var ]
    a_uu_p: Dict[ Tuple[str],gp.Var ]
    a_pw_p: Dict[ Tuple[str],gp.Var ]
    a_ps_p: Dict[ Tuple[str],gp.Var ]