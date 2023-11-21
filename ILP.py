from pulp import LpMaximize, LpMinimize, LpInteger, LpProblem, LpVariable, lpSum
import numpy as np

class ILP:
    def __init__(self, num_variables): # to check
        self.num_variables = num_variables
        
        self.model = LpProblem(name="ILP", sense=LpMaximize)

        self.variables = [LpVariable(
            f"x{i+1}", lowBound=0, cat=LpInteger) for i in range(num_variables)]
        self.set_objective(np.ones(num_variables), True)

    def add_constraint(self, row, constr_type, rhs):
        self.model += lpSum(row[i] * self.variables[i] for i in range(self.num_variables)) <= rhs if constr_type == 1 else \
            lpSum(row[i] * self.variables[i]
                  for i in range(self.num_variables)) >= rhs

    def add_constraints(self, rows, constr_type, rhs):
        for i in range(rows.shape[0]):
            self.add_constraint(rows[i, :], constr_type, rhs[i])

    def set_objective(self, row, is_maxim): # to check
        if is_maxim:
            self.model.sense = LpMaximize
        else:
            self.model.sense = LpMinimize
        self.model += lpSum(row[i] * self.variables[i]
                            for i in range(self.num_variables))

    def solve(self):
        self.model.solve()
        return self.model.status, [self.variables[i].varValue for i in range(self.num_variables)]
    
    def refresh(self):  # no idea
        pass

    def get_variables(self): # no idea
        pass
