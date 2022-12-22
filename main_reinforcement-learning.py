import argparse
import copy
from datetime import datetime
import json

from AbsObjetiveFunc import AbsObjetiveFunc
from SubstrateTree import SubstrateTree
from CRO_SL import CRO_SL
from CoralPopulation import Coral
import pdb
import time
import numpy as np

from erltrees.rl.configs import get_config
from erltrees.io import console, save_history_to_file
from erltrees.rl.utils import collect_metrics
from erltrees.evo.evo_tree import Individual
from erltrees.evo.utils import get_initial_pop

from rich import print
import VectorTree as vt
from utils import printv

def get_substrates_tree(cro_configs):
    substrates = []
    for substrate_tree in cro_configs["substrates_tree"]:
        substrates.append(SubstrateTree(substrate_tree["name"], substrate_tree["params"]))
    return substrates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRO-SL for Supervised Tree Induction')
    parser.add_argument('-t','--task',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c','--cro_config',help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-e','--episodes',help="How many episodes to use to evaluate an inividual?", required=True, type=int)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('-i', '--initial_pop', help="File with initial population", required=False, default='', type=str)
    parser.add_argument('--alpha',help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_jobs', help='How many jobs to parallelize?', required=False, default=-1, type=int)
    args = vars(parser.parse_args())

    depth = args["depth"]
    alpha = args["alpha"]
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)
    config = get_config(args["task"])

    n_attributes = config["n_attributes"]
    n_actions = config["n_actions"]
    gamma = (n_attributes - 1) * (2 ** (depth + 1)) / 2 - 1 
    
    command_line = str(args)
    command_line += "\n\npython main_reinforcement-learning.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = f"results/erltrees_log_{curr_time}.txt" 

    history = []
    for simulation in range(args['simulations']):
        console.rule(f"Simulation #{simulation}:")

        class ReinforcementLearningObjectiveFunc(AbsObjetiveFunc):
            def __init__(self, opt="max"):
                super().__init__(None, opt)

            def objetive(self, solution):
                collect_metrics(config, [solution], alpha=args["alpha"], episodes=args["episodes"],
                    should_norm_state=True, penalize_std=True, should_fill_attributes=True, n_jobs=args["n_jobs"])
                return solution.fitness
            
            def random_solution(self):
                solution = Individual.generate_random_tree(config, depth)
                return solution
            
            def check_bounds(self, solution):
                return solution.copy()

        objfunc = ReinforcementLearningObjectiveFunc()
        c = CRO_SL(objfunc, get_substrates_tree(cro_configs), cro_configs["general"])

        start_time = time.time()
        # Setting up initial population
        initial_pop = get_initial_pop(config, cro_configs["general"]["popSize"], args["depth"],
            alpha=args["alpha"], jobs_to_parallelize=-1, should_penalize_std=True, 
            should_norm_state=True, episodes=100, filename=args["initial_pop"])   

        c.population.population = [Coral(tree, objfunc=objfunc) for tree in initial_pop]

        _, fit = c.optimize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        # c.display_report()

        tree, _ = c.population.best_solution()
        collect_metrics(config, [tree], alpha=args["alpha"], episodes=1000,
            should_norm_state=True, penalize_std=True, should_fill_attributes=True)
        history.append((tree, tree.reward, tree.get_tree_size(), None))

        save_history_to_file(config, history, output_path, elapsed_time, command_line)
        print(f"Saved to '{output_path}'.")