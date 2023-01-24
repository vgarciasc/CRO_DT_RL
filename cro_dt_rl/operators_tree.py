import math
import random
import numpy as np

import math
import random
import numpy as np
import scipy as sp
import scipy.stats

import erltrees.evo.mutations as mut
from erltrees.evo.evo_tree import Individual

def random_mutation(solution):
    dice = np.random.randint(0, 9)

    if dice == 0:
        return expand_leaf(solution)
    elif dice == 1:
        return add_inner_node(solution)
    elif dice == 2:
        return truncate(solution)
    elif dice == 3:
        return replace_child(solution)
    elif dice == 4:
        return modify_leaf(solution)
    elif dice == 5:
        return modify_split(solution)
    elif dice == 6:
        return reset_split(solution)
    elif dice == 7:
        return cut_parent(solution)
    elif dice == 8:
        return prune_by_visits(solution)

## Mutation and recombination methods
def expand_leaf(solution: Individual):
    leaf = solution.get_random_node(get_inners=False, get_leaves=True)
    leaf.mutate_is_leaf()
    return solution

def add_inner_node(solution: Individual):
    node = solution.get_random_node(get_inners=True, get_leaves=False)
    node.mutate_add_inner_node()
    return solution

def truncate(solution: Individual):
    node = solution.get_random_node(get_inners=True, get_leaves=False)
    node.mutate_truncate_dx()
    return solution

def replace_child(solution: Individual):
    removal_depth_param = 1
    node_list = solution.get_node_list(get_inners=True, get_leaves=False)
    
    probabilities = [1 / (node.get_tree_size() ** removal_depth_param) for node in node_list]
    probabilities /= np.sum(probabilities)

    node = np.random.choice(node_list, p=probabilities)
    node.replace_child()
    return solution

def modify_leaf(solution: Individual):
    leaf = solution.get_random_node(get_inners=False, get_leaves=True)
    leaf.mutate_label()
    return solution

def modify_split(solution: Individual):
    node = solution.get_random_node(get_inners=True, get_leaves=False)
    node.mutate_threshold(0.1)
    return solution

def reset_split(solution: Individual):
    node = solution.get_random_node(get_inners=True, get_leaves=False)
    node.mutate_attribute()
    node.threshold = np.random.uniform(-1, 1)
    return solution

def cut_parent(solution: Individual):
    node = solution.get_random_node(get_inners=True, get_leaves=False)
    node.cut_parent()
    return solution

def crossover(solution: Individual, solution2: Individual):
    solution, _ = Individual.crossover(solution, solution2)
    return solution

def prune_by_visits(solution: Individual):
    solution.prune_by_visits()
    return solution