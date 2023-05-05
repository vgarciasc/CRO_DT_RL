import cro_dt_rl.cro_dt_rl as crodt
import json
import argparse
from datetime import datetime

from erltrees.rl.configs import get_config
from erltrees.io import console, save_history_to_file
from erltrees.rl.utils import collect_metrics
from erltrees.evo.evo_tree import Individual
from erltrees.evo.utils import get_initial_pop

import time
from rich import print

if __name__ == "__main__":
    # config = get_config("cartpole")
    # tree_str = "\n- Pole Angle <= 0.05429\n-- Cart Velocity <= -0.47464\n--- RIGHT\n--- LEFT\n-- RIGHT"
    config = get_config("lunar_lander")
    tree_str = "\n- Leg 1 is Touching <= 0.75000\n-- Y Velocity <= -0.08990\n--- Angle <= -0.03936\n---- Angular Velocity <= -0.14455\n----- LEFT ENGINE\n----- X Velocity <= 0.04595\n------ MAIN ENGINE\n------ Angular Velocity <= 0.25165\n------- LEFT ENGINE\n------- MAIN ENGINE\n---- Y Position <= 0.19867\n----- MAIN ENGINE\n----- Y Velocity <= -0.25715\n------ X Velocity <= -0.15895\n------- RIGHT ENGINE\n------- MAIN ENGINE\n------ X Velocity <= -0.04940\n------- RIGHT ENGINE\n------- Angle <= 0.16823\n-------- Angular Velocity <= 0.11400\n--------- LEFT ENGINE\n--------- RIGHT ENGINE\n-------- RIGHT ENGINE\n--- X Velocity <= -0.01815\n---- Angular Velocity <= -0.33115\n----- LEFT ENGINE\n----- RIGHT ENGINE\n---- Angular Velocity <= -0.00485\n----- LEFT ENGINE\n----- RIGHT ENGINE\n-- Angle <= -0.17938\n--- LEFT ENGINE\n--- Angle <= 0.18856\n---- Y Velocity <= -0.05735\n----- MAIN ENGINE\n----- NOP\n---- RIGHT ENGINE"

    cro_configs = json.load(open("configs/simple_erl_test.json", 'r'))
    results = {}

    initial_pop = [Individual.read_from_string(config, tree_str) for _ in range(10)]
    # for tree in initial_pop:
    #     tree.denormalize_thresholds()

    tree, c = crodt.run_cro_dt_rl(config, cro_configs, alpha=1.0, episodes=50,
                                  depth_random_indiv=2, n_jobs=-1,
                                  should_norm_state=True,
                                  task_solution_threshold=config["task_solution_threshold"],
                                  command_line="", output_path_temp="tmp.txt",
                                  initial_pop=initial_pop)
    results["basic_cro_dt_rl"] = tree.elapsed_time

    for k, v in results.items():
        print(f"{k}: {v:.4f} seconds")