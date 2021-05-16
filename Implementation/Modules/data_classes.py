from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
import gurobipy as gp

# Data Import Classes
@dataclass(frozen=True)
class ppe_data_class:
    ppe_type: str
    expected_units: int
    deviation: List[int]

@dataclass(frozen=True)
class model_param_class:
    cw: float
    cc: float
    M: float
    gamma: float
@dataclass()
class input_data_class:
    indices: Dict[ str, List[int] ]
    ppe_data: Dict[ str,ppe_data_class ]
    usage: Dict[ Tuple[str], List[int] ]
    arrival: Dict[ Tuple[str], List[int] ]
    transition: Dict[ Tuple[str], List[int] ]
    model_param: model_param_class
    expected_state_values: Dict[ str, Dict[Tuple[str], float] ]

# Data Export Classes

# State Action 
@dataclass(frozen=True)
class state:
    ue_tp: Dict[ Tuple[str],float ] # units expected on time t, ppe p
    uu_tp: Dict[ Tuple[str],float ] # units used on time t, ppe p
    pw_mdc: Dict[ Tuple[str],float ] # patients waiting for m periods, of complexity d, cpu c
    ps_tmdc: Dict[ Tuple[str],float ] # patients scheduled into time t, who have waited for m periods, of complexity d, cpu c
@dataclass(frozen=True)
class action:
    sc_tmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods to schedule into t (m of 0 corresponds to pe)
    rsc_ttpmdc: Dict[ Tuple[str],float ] # patients of complexity d, cpu c, waiting for m periods, to reschedule from t to tp 
    uv_tp: Dict[ Tuple[str],float ] # unit violations on time t, ppe p
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
    s_ue: Dict[ Tuple[str],gp.Var ]
    s_uu: Dict[ Tuple[str],gp.Var ]
    s_pw: Dict[ Tuple[str],gp.Var ]
    s_ps: Dict[ Tuple[str],gp.Var ]
    a_sc: Dict[ Tuple[str],gp.Var ]
    a_rsc: Dict[ Tuple[str],gp.Var ]
    a_uv: Dict[ Tuple[str],gp.Var ]
    a_uu_p: Dict[ Tuple[str],gp.Var ]
    a_pw_p: Dict[ Tuple[str],gp.Var ]
    a_ps_p: Dict[ Tuple[str],gp.Var ]