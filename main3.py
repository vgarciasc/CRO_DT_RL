import argparse
import copy
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
from SubstrateReal import SubstrateReal
from sup_configs import get_config, load_dataset
import pdb
import time
import numpy as np

from rich import print
import VectorTree as vt
from utils import printv

def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix
    string_full = prefix

    for config, history in zip(configs, histories):
        elapsed_times, multiv_info, univ_info = zip(*history)
        multiv_acc_in, multiv_acc_test, multiv_W, multiv_labels = zip(*multiv_info)
        univ_acc_in, univ_acc_test, univ_W, univ_labels = zip(*univ_info)

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"Average in-sample multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_in))} ± {'{:.3f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average in-sample univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_in))} ± {'{:.3f}'.format(np.std(univ_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_test))} ± {'{:.3f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += f"Average test univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_test))} ± {'{:.3f}'.format(np.std(univ_acc_test))}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += f"DATASET: {config['name']}\n"
        for (elapsed_time, \
            (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
            (univ_acc_in, univ_acc_test, univ_labels, univ_W)) in history:

            string_full += f"In-sample:" + "\n"
            string_full += f"  Multivariate accuracy: {multiv_acc_in}" + "\n"
            string_full += f"  Univariate accuracy: {univ_acc_in}" + "\n"
            string_full += f"Out-of-sample:" + "\n"
            string_full += f"  Multivariate accuracy: {multiv_acc_test}" + "\n"
            string_full += f"  Univariate accuracy: {univ_acc_test}" + "\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            string_full += "Multivariate tree:\n" + vt.weights2treestr(multiv_W, multiv_labels, config) + "\n"
            string_full += "Univariate tree:\n" + vt.weights2treestr(univ_W, univ_labels, config) + "\n"
            string_full += "\n\n--------\n\n"

    with open(output_path_summary, "w", encoding="utf-8") as text_file:
        text_file.write(string_summ)

    with open(output_path_full, "w", encoding="utf-8") as text_file:
        text_file.write(string_full)

def get_substrates_real(cro_configs):
    substrates = []
    for substrate_real in cro_configs["substrates_real"]:
        substrates.append(SubstrateReal(substrate_real["name"], substrate_real["params"]))
    return substrates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRO-SL for Supervised Tree Induction')
    parser.add_argument('-i','--dataset',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c','--cro_config',help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--alpha',help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    command_line = str(args)
    command_line += "\n\npython main3.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    output_path_summ = f"results/log_{curr_time}_summary.txt"
    output_path_full = f"results/log_{curr_time}_full.txt"

    depth = args["depth"]
    alpha = args["alpha"]
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)

    if args['dataset'] == 'all':
        data_configs = [
            get_config("breast_cancer"),
            get_config("car"),
            get_config("banknote"),
            get_config("balance"),
            get_config("acute-1"),
            get_config("acute-2"),
            get_config("transfusion"),
            get_config("climate"),
            get_config("sonar"),
            get_config("optical"),
        ]
    else:
        data_configs = [get_config(args['dataset'])]

    histories = []
    for data_config in data_configs:
        n_attributes = data_config["n_attributes"]

        histories.append([])
        for iteration in range(args['simulations']):
            print(f"Iteration #{iteration}:")
            X, y = load_dataset(data_config)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=iteration)

            class SupervisedObjectiveFunc(AbsObjetiveFunc):
                def __init__(self, size, opt="max"):
                    self.size = size
                    super().__init__(self.size, opt)

                def objetive(self, solution):
                    W = solution.reshape((2**depth - 1, n_attributes + 1))
                    accuracy, _ = vt.get_accuracy(X_train, y_train, W)
                    
                    # return accuracy

                    penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(W)])

                    # b = np.abs(W)
                    # b[np.arange(len(b)), np.argmax(b[:, 1:], axis=1) + 1] = 0
                    # penalty = np.sum(np.abs(W) - b)
                    
                    # alpha = 2

                    return accuracy - alpha * penalty
                
                def random_solution(self):
                    return vt.generate_random_weights(n_attributes, depth)
                
                def check_bounds(self, solution):
                    return solution

            start_time = time.time()
            sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
            c = CRO_SL(SupervisedObjectiveFunc(sol_size), get_substrates_real(cro_configs), cro_configs["general"])
            _, fit = c.optimize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            # c.display_report()

            multiv_W, _ = c.population.best_solution()
            multiv_W = multiv_W.reshape((2**depth - 1, n_attributes + 1))

            b = np.copy(multiv_W)
            b2 = np.abs(multiv_W)
            b[np.arange(len(b)), np.argmax(b2[:,1:], axis=1) + 1] = 0
            b[:, 0] = 0
            univ_W = multiv_W - b
            
            _, multiv_labels = vt.get_accuracy(X_train, y_train, multiv_W)
            _, univ_labels = vt.get_accuracy(X_train, y_train, univ_W)
            y_pred_train = vt.predict_batch(X_train, multiv_W, multiv_labels, add_1=True)
            y_pred_test = vt.predict_batch(X_test, multiv_W, multiv_labels, add_1=True)
            multiv_acc_in = np.mean([(1 if y_pred_i == y_train_i else 0) for y_pred_i, y_train_i in zip(y_train, y_pred_train)])
            multiv_acc_test = np.mean([(1 if y_pred_i == y_train_i else 0) for y_pred_i, y_train_i in zip(y_test, y_pred_test)])
            
            univ_acc_in, univ_labels_in = vt.get_accuracy(X_train, y_train, univ_W)
            univ_acc_test, univ_labels_test = vt.get_accuracy(X_test, y_test, univ_W)

            histories[-1].append((elapsed_time, \
                (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
                (univ_acc_in, univ_acc_test, univ_labels, univ_W)))
            save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)
        
        print(f"Saved to '{output_path_summ}'.")
        print(f"Saved to '{output_path_full}'.")