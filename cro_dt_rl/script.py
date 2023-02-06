import argparse
import json
import time
from datetime import datetime
import pdb
import numpy as np
from copy import deepcopy
from erltrees.rl.configs import get_config
from erltrees.il.dagger import run_dagger
from erltrees.experiments.reward_pruning import reward_pruning
import erltrees.rl.utils as rl
from erltrees.evo.evo_tree import Individual
from cro_dt_rl.cro_dt_rl import run_cro_dt_rl

from rich import print, get_console

def save_to_file(command_line, filename, history):
    string = ""
    string += command_line + "\n\n"
    string += "=" * 50 + "\n\n"

    daggers, rps, cros = zip(*history)
    if getattr(daggers[0], "get_size", None) is not None:
        string += f"Average DAgger: (reward: {'{:.3f}'.format(np.mean([d.reward for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.mean([d.std_reward for d in daggers if d is not None]))}, success_rate: {'{:.3f}'.format(np.mean([d.success_rate for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.std([d.success_rate for d in daggers if d is not None]))}, tree size: {'{:.3f}'.format(np.mean([d.get_size() for d in daggers if d is not None]))}, elapsed_time: {'{:.3f}'.format(np.mean([d.elapsed_time for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.std([d.reward for d in daggers if d is not None]))} \n"
    else:
        string += f"Average DAgger: (reward: {'{:.3f}'.format(np.mean([d.reward for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.mean([d.std_reward for d in daggers if d is not None]))}, success_rate: {'{:.3f}'.format(np.mean([d.success_rate for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.std([d.success_rate for d in daggers if d is not None]))}, tree size: {'{:.3f}'.format(np.mean([d.get_tree_size() for d in daggers if d is not None]))}, elapsed_time: {'{:.3f}'.format(np.mean([d.elapsed_time for d in daggers if d is not None]))} ± {'{:.3f}'.format(np.std([d.reward for d in daggers if d is not None]))} \n"
    string += f"Average Reward Pruning: (reward: {'{:.3f}'.format(np.mean([d.reward for d in rps if d is not None]))} ± {'{:.3f}'.format(np.mean([d.std_reward for d in rps if d is not None]))}, success_rate: {'{:.3f}'.format(np.mean([d.success_rate for d in rps if d is not None]))} ± {'{:.3f}'.format(np.std([d.success_rate for d in rps if d is not None]))}, tree size: {'{:.3f}'.format(np.mean([d.get_tree_size() for d in rps if d is not None]))}, elapsed_time: {'{:.3f}'.format(np.mean([d.elapsed_time for d in rps if d is not None]))} ± {'{:.3f}'.format(np.std([d.reward for d in rps if d is not None]))} \n"
    string += f"Average CRO-DT-RL: (reward: {'{:.3f}'.format(np.mean([d.reward for d in cros if d is not None]))} ± {'{:.3f}'.format(np.mean([d.std_reward for d in cros if d is not None]))}, success_rate: {'{:.3f}'.format(np.mean([d.success_rate for d in cros if d is not None]))} ± {'{:.3f}'.format(np.std([d.success_rate for d in cros if d is not None]))}, tree size: {'{:.3f}'.format(np.mean([d.get_tree_size() for d in cros if d is not None]))}, elapsed_time: {'{:.3f}'.format(np.mean([d.elapsed_time for d in cros if d is not None]))} ± {'{:.3f}'.format(np.std([d.reward for d in cros if d is not None]))} \n"
    string += "=" * 50 + "\n\n"

    for simulation, (dagger_tree, rp_tree, cro_tree) in enumerate(history):
        string += f"SIMULATION {simulation}\n"
        string += "=" * 50 + "\n"
        if getattr(daggers[0], "get_size", None) is not None:
            string += f"DAgger (size: {dagger_tree.get_size()}, reward: {'{:.3f}'.format(dagger_tree.reward)} ± {'{:.3f}'.format(dagger_tree.std_reward)}, success rate: {'{:.3f}'.format(dagger_tree.success_rate)}, elapsed_time: {'{:.3f}'.format(dagger_tree.elapsed_time)}): \n"
            string += "-" * 50 + "\n"
            string += dagger_tree.get_as_viztree()
        else:
            string += f"DAgger (size: {dagger_tree.get_tree_size()}, reward: {'{:.3f}'.format(dagger_tree.reward)} ± {'{:.3f}'.format(dagger_tree.std_reward)}, success rate: {'{:.3f}'.format(dagger_tree.success_rate)}, elapsed_time: {'{:.3f}'.format(dagger_tree.elapsed_time)}): \n"
            string += "-" * 50 + "\n"
            string += str(dagger_tree)

        string += "\n\n"

        if rp_tree is not None:
            string += f"Reward Pruning (size: {rp_tree.get_tree_size()}, reward: {'{:.3f}'.format(rp_tree.reward)} ± {'{:.3f}'.format(rp_tree.std_reward)}, success rate: {'{:.3f}'.format(rp_tree.success_rate)}, elapsed_time: {'{:.3f}'.format(rp_tree.elapsed_time)}): \n"
            string += "-" * 50 + "\n"
            string += str(rp_tree)
            string += "\n\n"

        if cro_tree is not None:
            string += f"CRO-DT-RL (size: {cro_tree.get_tree_size()}, reward: {'{:.3f}'.format(cro_tree.reward)} ± {'{:.3f}'.format(cro_tree.std_reward)}, success rate: {'{:.3f}'.format(cro_tree.success_rate)}, elapsed_time: {'{:.3f}'.format(cro_tree.elapsed_time)}): \n"
            string += "-" * 50 + "\n"
            string += str(cro_tree)
            string += "\n\n"

    with open(filename, "w") as file:
        file.write(string)

    print(f"Saved log to '{filename}'.")

console = get_console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-e','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-a','--fitness_alpha', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-p','--pruning_alpha', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-j','--episodes', help='Number of episodes to collect every iteration', required=True, type=int)
    parser.add_argument('-c','--cro_config', help='CRO config filepath', required=True, type=str)
    parser.add_argument('--dagger_file', help='Should get DAgger trees from file?', required=False, default=None, type=str)
    parser.add_argument('--expert_exploration_rate', help='The epsilon to use during dataset collection', required=False, default=0.0, type=float)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=0, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_attenuate_alpha', help='Should attenuate alpha?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_penalize_std', help='Should penalize std?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--rp_alpha', help='Which alpha to use?', required=True, default=1.0, type=float)
    parser.add_argument('--rp_ks_test', help='Should use KS test to detect if trees are equal?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--rp_ks_threshold', help='Which KS test threshold to use?', required=False, default=0.1, type=float)
    parser.add_argument('--rp_rounds', help='How many rounds for reward pruning?', required=True, default=1, type=int)
    parser.add_argument('--rp_episodes', help='How many episodes to run in reward pruning?', required=True, type=int)
    parser.add_argument('--cro_episodes', help='How many episodes to run in CRO?', required=True, type=int)
    parser.add_argument('--simulations', help='How many simulations to run?', required=False, default=-1, type=int)
    parser.add_argument('--n_jobs', help='How many jobs to parallelize?', required=False, default=-1, type=int)
    parser.add_argument('--output_prefix',help='Which output name to use?', required=False, default="cro-dt-rl", type=str)
    args = vars(parser.parse_args())

    command_line = str(args)
    command_line += "\n\npython -m cro_dt_rl.cro_dt_rl " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = f"results/{args['output_prefix']}_{curr_time}.txt" 
    output_path_temp = f"results/{args['output_prefix']}_tmp_{curr_time}.txt" 

    config = get_config(args["task"])
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)
    
    command_line += f"\n\n {str(cro_configs)} \n\n"

    # Loading DAgger trees from file, if provided
    dagger_trees_from_file = None
    if args['dagger_file']:
        with open(args['dagger_file'], 'r') as f:
            dagger_trees_from_file = json.load(f)
    else:
        from erltrees.il.parser import handle_args
        expert, X, y = handle_args(args, config)

    history = []
    for simulation in range(args["simulations"]):
        start_time = time.time()
        # Imitation phase
        if dagger_trees_from_file is None:
            dagger_tree, _, _ = run_dagger(
                config, X, y,
                expert=expert, 
                model_name="DistilledTree",
                pruning_alpha=args['pruning_alpha'],
                fitness_alpha=args['fitness_alpha'],
                iterations=args['iterations'],
                episodes=args['episodes'],
                should_penalize_std=args['should_penalize_std'],
                task_solution_threshold=args['task_solution_threshold'],
                should_attenuate_alpha=args['should_attenuate_alpha'],
                n_jobs=args['n_jobs'])
        else:
            dagger_tree_idx = simulation % len(dagger_trees_from_file)
            dagger_tree = Individual.read_from_string(config, dagger_trees_from_file[dagger_tree_idx])
            print(f"[red]Using [yellow]DAgger tree #{dagger_tree_idx}[/yellow] (size: {dagger_tree.get_tree_size()})[/red]")

        elapsed_time = time.time() - start_time

        rl.collect_metrics(config, [dagger_tree], alpha=args['fitness_alpha'], episodes=1000,
            should_norm_state=False, penalize_std=True, should_fill_attributes=True,
            task_solution_threshold=args['task_solution_threshold'], n_jobs=args["n_jobs"])
        dagger_tree.elapsed_time = elapsed_time

        history.append((dagger_tree, None, None))
        save_to_file(command_line, output_path, history)

        if type(dagger_tree) is not Individual:
            rp_tree = Individual.read_from_string(config, dagger_tree.get_as_viztree())
        else:
            rp_tree = deepcopy(dagger_tree)
        
        # Pruning phase
        start_time = time.time()
        for round in range(args['rp_rounds']):
            console.rule(f"[red]Round {round}/{args['rp_rounds']} of reward pruning.")
            rp_tree, _ = reward_pruning(rp_tree, rp_tree, config, episodes=args['rp_episodes'],
                alpha=args['rp_alpha'], task_solution_threshold=args['task_solution_threshold'],
                should_norm_state=False, should_use_kstest=args['rp_ks_test'],
                kstest_threshold=args['rp_ks_threshold'], n_jobs=args['n_jobs'], verbose=True)
        elapsed_time = time.time() - start_time
        rp_tree.elapsed_time = elapsed_time

        rl.collect_metrics(config, [rp_tree], alpha=args['fitness_alpha'], episodes=1000,
            should_norm_state=False, penalize_std=True, should_fill_attributes=True,
            task_solution_threshold=args['task_solution_threshold'], n_jobs=args["n_jobs"])

        history[-1] = (dagger_tree, rp_tree, None)
        save_to_file(command_line, output_path, history)

        rp_as_initial = deepcopy(rp_tree)
        rp_as_initial.normalize_thresholds()

        # Fine-tuning phase
        cro_tree, c = run_cro_dt_rl(config, cro_configs, args['fitness_alpha'],
            should_norm_state=True, episodes=args['cro_episodes'], 
            initial_pop=[rp_as_initial], n_jobs=args['n_jobs'],
            command_line=command_line, output_path_temp=output_path_temp)

        rl.collect_metrics(config, [cro_tree], alpha=args['fitness_alpha'], episodes=1000,
            should_norm_state=True, penalize_std=True, should_fill_attributes=True,
            task_solution_threshold=args['task_solution_threshold'], n_jobs=args["n_jobs"])

        history[-1] = (dagger_tree, rp_tree, cro_tree)
        save_to_file(command_line, output_path, history)