import random

import pdb
import numpy as np
from cro_dt_rl.operators_tree import *
from Substrate import *

"""
Substrate class that has operations for handling RL trees
"""
class SubstrateTree(Substrate):
    def __init__(self, evolution_method, params = None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)
    
    """
    Evolves a solution with a different strategy depending on the type of operator
    """
    def evolve(self, solution, population, objfunc):
        result = None
        
        others = [i for i in population if i != solution]
        if len(others) > 1:
            solution2 = random.choice(others)
        else:
            solution2 = solution

        if self.evolution_method == "expand_leaf":
            result = expand_leaf(solution.solution.copy())
        elif self.evolution_method == "expand_leaf_continuous":
            result = expand_leaf_continuous(solution.solution.copy())
        elif self.evolution_method == "add_inner_node":
            result = add_inner_node(solution.solution.copy())
        elif self.evolution_method == "truncate":
            result = truncate(solution.solution.copy())
        elif self.evolution_method == "replace_child":
            result = replace_child(solution.solution.copy())
        elif self.evolution_method == "modify_leaf":
            result = modify_leaf(solution.solution.copy())
        elif self.evolution_method == "modify_leaf_continuous":
            result = modify_leaf_continuous(solution.solution.copy())
        elif self.evolution_method == "modify_split":
            result = modify_split(solution.solution.copy())
        elif self.evolution_method == "reset_split":
            result = reset_split(solution.solution.copy())
        elif self.evolution_method == "prune_by_visits":
            result = prune_by_visits(solution.solution.copy())
        elif self.evolution_method == "cut_parent":
            result = cut_parent(solution.solution.copy())
        elif self.evolution_method == "crossover":
            result = crossover(solution.solution.copy(), solution2.solution.copy())
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)
            
        return result
