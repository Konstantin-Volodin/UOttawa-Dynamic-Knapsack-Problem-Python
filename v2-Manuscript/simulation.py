# %%

import numpy as np
from gurobipy import *

gamma = 0.99
states = ['s0', 's1', 's2']

actions = {
    's0': ['a0', 'a1'],
    's1': ['a0', 'a1'],
    's2': ['a0', 'a1'],
}

data = {
    ('s0', 'a0'): {'transition': [0.5, 0, 0.5],     'cost': [0, 0, 0]},
    ('s0', 'a1'): {'transition': [0, 0, 1],         'cost': [0, 0, 0]},
    ('s1', 'a0'): {'transition': [0.7, 0.1, 0.2],   'cost': [5, 0, 0]},
    ('s1', 'a1'): {'transition': [0, 0.95, 0.05],   'cost': [0, 0, 0]},
    ('s2', 'a0'): {'transition': [0.4, 0, 0.6],     'cost': [0, 0, 0]},
    ('s2', 'a1'): {'transition': [0.3, 0.3, 0.4],   'cost': [-1, 0, 0]}
}


# %% Optimization Model
model = Model('MDP')

# Variables
var = model.addVars(states)

# Constraints
constr = model.addConstrs(
    np.dot(data[(sa, ac)]['cost'], data[(sa, ac)]['transition'])
    + gamma * quicksum(var[states[d]]*data[(sa, ac)]['transition'][d] for d in range(len(states)))
    <= var[sa]
    for sa in states
    for ac in actions[sa]
)

# Objective Function
model.setObjective(var.sum(), GRB.MINIMIZE)
model.optimize()
values = []
for i in var:
    values.append(var[i].x)

# %% Generate Policy

policy = {}
for sa in states:
    model = Model('Policy')

    var = model.addVars(actions[sa], vtype=GRB.BINARY)
    model.addConstr(var.sum() == 1)

    obj_expr = LinExpr()
    for ac in var:
        cost = np.dot(data[(sa, ac)]['cost'],data[(sa, ac)]['transition'])
        cost += np.dot(values,data[(sa, ac)]['transition'])

        obj_expr.addTerms(cost, var[ac])
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.optimize()

    for i in var:
        if var[i].X >= 0.5:
            policy[sa] = i
            break

# %% Simulation


