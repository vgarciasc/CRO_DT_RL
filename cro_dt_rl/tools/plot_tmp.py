import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot from tmp')
    parser.add_argument('-f','--filename',help='Which file to plot?', required=True, type=str)
    parser.add_argument('-s','--task_solution',help='What is the task solution?', required=True, type=str)
    args = vars(parser.parse_args())

    simulations = []
    with open(args['filename'], 'r') as f:
        for line in f.readlines():
            if "Generation #1 " in line:
                simulations.append([])
            if "Reward" in line:
                content = line.split(" ")
                avg_reward = float(content[1])
                std_reward = float(content[3][:-1])
                tree_size = int(content[5][:-1])
                simulations[-1].append((avg_reward, std_reward, tree_size))
    
    for i, simulation in enumerate(simulations):
        avg_rewards, std_rewards, tree_sizes = zip(*simulation)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 6))
        x = range(len(avg_rewards))
        # ax1.axhline(y=args['task_solution'], color='black', linestyle='--')
        ax1.plot(x, avg_rewards, label="Reward", color="blue")
        ax1.fill_between(x, 
            np.array(avg_rewards) - np.array(std_rewards),
            np.array(avg_rewards) + np.array(std_rewards),
            color="blue", alpha=0.2)
        ax2.plot(range(len(avg_rewards)), tree_sizes, label="Tree size", color="red")
        ax2.set_xlabel("Generations")
        ax1.set_ylabel("Reward")
        ax2.set_ylabel("Tree size")
        plt.suptitle(args['filename'] + f"\nSimulation {i}")
        plt.savefig(args['filename'].split(".txt")[0] + f"_simul{i}.png")