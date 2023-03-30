import argparse
import copy
from datetime import datetime
import json

from AbsObjetiveFunc import AbsObjetiveFunc
from cro_dt_rl.SubstrateTree import SubstrateTree
from cro_dt_rl.utils import printv
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


def get_substrates_tree(cro_configs):
    substrates = []
    for substrate_tree in cro_configs["substrates_tree"]:
        substrates.append(SubstrateTree(substrate_tree["name"], substrate_tree["params"]))
    return substrates


def save_info(cro_sl, gen):
    best_tree, _ = cro_sl.population.best_solution()

    if gen == 0:
        mode = "w"
    else:
        mode = "a"

    with open(cro_sl.save_info_filename, mode) as file:
        string = ""
        if gen == 1:
            string += cro_sl.command_line
            string += "\n\n"

        string += f"Generation #{gen} \n"
        string += f"Reward {best_tree.reward} +- {best_tree.std_reward}, Size {best_tree.get_tree_size()}, Success Rate {best_tree.success_rate} \n"
        string += "-" * 50 + "\n"
        string += str(best_tree)
        string += "\n\n"

        file.write(string)


def run_cro_dt_rl(config, cro_configs, alpha, episodes,
                  should_norm_state=False, should_penalize_std=True,
                  depth_random_indiv=3, initial_pop=None,
                  task_solution_threshold=-1,
                  command_line="", output_path_temp="tmp.txt",
                  n_jobs=-1):
    class ReinforcementLearningObjectiveFunc(AbsObjetiveFunc):
        def __init__(self, opt="max"):
            super().__init__(None, opt)

        def objetive(self, solution):
            collect_metrics(config, [solution], alpha=alpha, episodes=episodes,
                            should_norm_state=should_norm_state, penalize_std=should_penalize_std,
                            task_solution_threshold=task_solution_threshold,
                            should_fill_attributes=True, n_jobs=n_jobs)
            return solution.fitness

        def random_solution(self):
            solution = Individual.generate_random_tree(config, depth_random_indiv)
            return solution

        def check_bounds(self, solution):
            return solution.copy()

    objfunc = ReinforcementLearningObjectiveFunc()
    c = CRO_SL(objfunc, get_substrates_tree(cro_configs), cro_configs["general"])

    c.config = config
    c.alpha = alpha
    c.episodes = episodes
    c.should_norm_state = should_norm_state
    c.should_penalize_std = should_penalize_std
    c.task_solution_threshold = task_solution_threshold
    c.n_jobs = n_jobs

    c.save_info = save_info
    c.save_info_filename = output_path_temp
    c.command_line = command_line

    # Setting up initial population
    initial_pop = get_initial_pop(config, alpha=alpha,
                                  popsize=cro_configs["general"]["popSize"],
                                  depth_random_indiv=depth_random_indiv, n_jobs=n_jobs,
                                  should_penalize_std=True, should_norm_state=should_norm_state,
                                  episodes=100, initial_pop=initial_pop)

    c.population.population = []
    for tree in initial_pop:
        coral = Coral(tree, objfunc=objfunc)
        coral.get_fitness()
        c.population.population.append(coral)

    # Running optimization
    start_time = time.time()
    _, fit = c.optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    tree, _ = c.population.best_solution()
    tree.elapsed_time = elapsed_time

    return tree, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCRO-SL for Supervised Tree Induction')
    parser.add_argument('-t', '--task', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c', '--cro_config', help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-e', '--episodes', help="How many episodes to use to evaluate an inividual?", required=True, type=int)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-d', '--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('-i', '--initial_pop', help="File with initial population", required=False, default=None, type=str)
    parser.add_argument('--start_from_idx', help="Start from idx", required=False, default=0, type=int)
    parser.add_argument('--initial_pop_individual', help="Should iterate over initial pop file to use individuals as starting population?", required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_norm_state', help="Should normalize state?", required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=True, default=None, type=int)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="cro-dt-rl", type=str)
    parser.add_argument('--alpha', help="How to penalize tree multivariateness?", required=True, type=float)
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
    command_line += "\n\npython -m cro_dt_rl.cro_dt_rl " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = f"results/{args['output_prefix']}_{curr_time}.txt"
    output_path_temp = f"results/{args['output_prefix']}_tmp_{curr_time}.txt"

    if args['initial_pop']:
        with open(args['initial_pop']) as f:
            json_obj = json.load(f)
        initial_pop = [Individual.read_from_string(config, json_str) for json_str in json_obj]

    history = []
    simulation = args['start_from_idx']
    while simulation < args['simulations']:
        console.rule(f"[red]Simulation #{simulation} / {args['simulations']} [/red]:")

        if args['initial_pop']:
            if args['initial_pop_individual']:
                initial_pop_now = [initial_pop[simulation]]
            else:
                initial_pop_now = initial_pop
        else:
            initial_pop_now = None

        tree, c = run_cro_dt_rl(config, cro_configs, alpha, args['episodes'],
                                depth_random_indiv=depth, n_jobs=args['n_jobs'],
                                should_norm_state=args['should_norm_state'],
                                task_solution_threshold=args['task_solution_threshold'],
                                command_line=command_line, output_path_temp=output_path_temp,
                                initial_pop=initial_pop_now)

        collect_metrics(config, [tree], alpha=args["alpha"], episodes=1000, should_norm_state=args['should_norm_state'],
                        penalize_std=True, should_fill_attributes=True, task_solution_threshold=config['task_solution_threshold'],)
        history.append((tree, tree.reward, tree.std_reward, tree.get_tree_size(), tree.success_rate))

        save_history_to_file(config, history, output_path, None, command_line)
        print(f"Saved to '{output_path}'.")

        simulation += 1
