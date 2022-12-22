import random

import numpy as np
from operators_soft_tree import *
from Substrate import *

"""
Substrate class that has continuous mutation and cross methods
"""
class SubstrateSoftTree(Substrate):
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
        
        if self.evolution_method == "DE/rand/1":
            result = DERand1(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/best/1":
            result = DEBest1(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/rand/2":
            result = DERand2(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/best/2":
            result = DEBest2(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/current-to-rand/1":
            result = DECurrentToRand1(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/current-to-best/1":
            result = DECurrentToBest1(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        elif self.evolution_method == "DE/current-to-pbest/1":
            result = DECurrentToPBest1(solution.solution.copy(), others, self.params["F"], self.params["Cr"])
        else:
            print(f"Error: evolution method \"{self.evolution_method}\" not defined")
            exit(1)
            
        
        return result
