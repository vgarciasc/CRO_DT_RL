import random

import numpy as np
from operators_discrete_tree import *
from Substrate import *

"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateDiscreteTree(Substrate):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Evolves a solution with a different strategy depending on the type of operator
    """
    def evolve(self, solution, population, objfunc):
        result = None
        
        if self.evolution_method == "switch_attribute":
            result = switch_attribute(solution.solution.copy(), objfunc.n_attributes, self.params["F"])
        elif self.evolution_method == "modify_threshold":
            result = modify_threshold(solution.solution.copy(), self.params["F"])
        elif self.evolution_method == "reset_threshold":
            result = reset_threshold(solution.solution.copy())
        elif self.evolution_method == "reset_split":
            result = reset_split(solution.solution.copy(), objfunc.n_attributes)
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)
            
        
        return result
