import math
import argparse
import copy
from datetime import datetime
from SoftTree import SoftTree
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
from CoralPopulation import Coral
from SubstrateSoftTree import SubstrateSoftTree
from sup_configs import get_config, load_dataset
import pdb
import time
import numpy as np

from sklearn.preprocessing import StandardScaler

from rich import print
import VectorTree as vt
from utils import printv
from cart import get_cart_as_W

def get_initial_pop(data_config, popsize, X_train, y_train,  
    should_cart_init, desired_depth, cart_pop_filepath, objfunc):

    # Creating CART population
    cart_pop = []
    if cart_pop_filepath != None:
        with open(cart_pop_filepath, "rb") as f:
            cart_pops = pickle.load(f)
        
        for (dataset_code, pop) in cart_pops:
            if dataset_code == data_config["code"]:
                cart_pop = pop
    elif should_cart_init:
        for alpha in np.linspace(0, 0.5, 1000):
            for criterion in ["gini", "entropy", "log_loss"]:
                dt = DecisionTreeClassifier(ccp_alpha=alpha, criterion=criterion)
                dt.fit(X_train, y_train)

                if dt.get_depth() == desired_depth:
                    W = get_cart_as_W(data_config, dt, desired_depth)
                    cart_pop.append(W)

        cart_pop = np.unique(cart_pop, axis=0)
        cart_pop = [f for f in cart_pop]

        for _ in range(len(cart_pop), popsize // 3):
            dt = DecisionTreeClassifier(max_depth=desired_depth, splitter="random")
            dt.fit(X_train, y_train)
            W = get_cart_as_W(data_config, dt, desired_depth)
            cart_pop.append(W)

        if len(cart_pop) > popsize // 3:
            cart_pop = cart_pop[:popsize // 3]
    
    # Creating mutated CART population
    mutated_cart_pop = []
    for cart in cart_pop:
        mutated_cart = cart + np.random.normal(0, 1, size=cart.shape)
        mutated_cart_pop.append(mutated_cart)

    # Creating random population based on vectors
    random_pop_continuous = []
    for _ in range(len(cart_pop) + len(mutated_cart_pop), popsize):
        random_pop_continuous.append(objfunc.random_solution())

    return cart_pop + mutated_cart_pop + random_pop_continuous

def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix + "\n"
    string_full = prefix + "\n"

    for config, history in zip(configs, histories):
        elapsed_times, trees, accs = zip(*history)
        acc_in, acc_test = zip(*accs)

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"{len(elapsed_times)} simulations executed.\n"
        string_summ += f"Average in-sample accuracy: {'{:.3f}'.format(np.mean(acc_in))} ± {'{:.3f}'.format(np.std(acc_in))}\n"
        string_summ += f"Average test accuracy: {'{:.3f}'.format(np.mean(acc_test))} ± {'{:.3f}'.format(np.std(acc_test))}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += "--------------------------------------------------\n\n"
        string_full += f"DATASET: {config['name']}\n"
        
        for (elapsed_time, tree, \
            (univ_acc_in, univ_acc_test)) in history:

            string_full += f"In-sample:" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_in}" + "\n"
            string_full += f"Test:" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_test}" + "\n"
            # if scaler is not None:
            #     string_full += f"Scaler: (mean, {scaler.mean_})\n"
            #     string_full += f"        (var,  {scaler.var_})\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            string_full += "\n--------\n"
            string_full += str(tree)
            string_full += "\n\n--------\n\n"

    with open(output_path_summary, "w", encoding="utf-8") as text_file:
        text_file.write(string_summ)

    with open(output_path_full, "w", encoding="utf-8") as text_file:
        text_file.write(string_full)

def get_substrates_real(cro_configs):
    substrates = []
    for substrate_real in cro_configs["substrates_real"]:
        substrates.append(SubstrateSoftTree(substrate_real["name"], substrate_real["params"]))
    return substrates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRO-SL for Supervised Tree Induction')
    parser.add_argument('-i','--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c','--cro_config', help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-s','--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--initial_pop', help="File with initial population", required=False, default=None, type=str)
    parser.add_argument('--alpha', help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--should_normalize_rows', help='Should normalize rows?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_cart_init', help='Should initialize with CART trees?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_normalize_penalty', help='Should normalize penalty?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_apply_exponential', help='Should apply exponential penalty?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_use_threshold', help='Should ignore weights under a certain threshold?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--threshold', help="Under which threshold should weights be ignored?", required=False, default=0.05, type=float)
    parser.add_argument('--should_use_univariate_accuracy', help='Should use univariate tree\'s accuracy when measuring fitness?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--start_from', help='Should start from where?', required=False, default=0, type=int)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="log", type=str)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    depth = args["depth"]
    alpha = args["alpha"]
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)
    popsize = cro_configs["general"]["popSize"]

    command_line = str(args)
    command_line += "\n\npython main_ST.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path_summ = f"results/{args['output_prefix']}_{curr_time}_summary.txt"
    output_path_full = f"results/{args['output_prefix']}_{curr_time}_full.txt"
    
    normal_dataset_list = ["breast_cancer", "car", "banknote", "balance", "acute-1", "acute-2", "transfusion", "climate", "sonar", "optical", "drybean", "avila", "wine-red", "wine-white"]
    artificial_dataset_list = ["artificial_100_3_2", "artificial_1000_3_2", "artificial_1000_3_10", "artificial_1000_10_10", "artificial_10000_3_10", "artificial_10000_3_10"]

    if args['dataset'].startswith("artificial"):
        dataset_list = artificial_dataset_list
    else:
        dataset_list = normal_dataset_list

    if args['dataset'].endswith('all'):
        data_configs = [get_config(d) for d in dataset_list]
    elif args['dataset'].endswith("onwards"):
        dataset_start = dataset_list.index(args['dataset'][:-len("_onwards")])
        data_configs = [get_config(d) for d in dataset_list[dataset_start:]]
    else:
        data_configs = [get_config(args['dataset'])]

    histories = []
    for data_config in data_configs:
        n_attributes = data_config["n_attributes"]
        n_classes = data_config["n_classes"]

        histories.append([])
        
        start_idx = args['start_from'] if data_config == data_configs[0] else 0
        simulations = range(args["simulations"])[start_idx:]

        for simulation in simulations:
            X, y = load_dataset(data_config)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation, stratify=y)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation, stratify=y_test)
            
            if args["should_normalize_dataset"]:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                # X_val = scaler.transform(X_val)
            else:
                scaler = None

            print(f"Iteration #{simulation}:")

            class SupervisedObjectiveFunc(AbsObjetiveFunc):
                def __init__(self, size, opt="max"):
                    self.size = size
                    super().__init__(self.size, opt)

                def objetive(self, solution):
                    if args["should_use_univariate_accuracy"]:
                        solution.turn_univariate()
                    
                    solution.update_leaves_by_dataset(X_train, y_train)
                    y_pred = solution.predict_batch(X_train)
                    accuracy = np.mean([(1 if y_pred[i] == y_train[i] else 0) for i in range(len(X_train))])
                    return accuracy

                    # accuracy, _ = solution.dt_matrix_fit(X_train, y_train)
                    # return accuracy
                
                def random_solution(self):
                    solution = SoftTree(n_attributes, n_classes, depth)
                    solution.randomize(depth)
                    solution.update_leaves_by_dataset(X_train, y_train)
                    solution.X_ = np.vstack((np.ones(len(X_train)).T, X_train.T)).T
                    solution.Y_ = np.tile(y_train + 42, (2 ** depth, 1))
                    return solution
                
                def check_bounds(self, solution):
                    return solution

            sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
            objfunc = SupervisedObjectiveFunc(sol_size)

            c = CRO_SL(objfunc, get_substrates_real(cro_configs), cro_configs["general"])
            
            start_time = time.time()
            _, fit = c.optimize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            # c.display_report()

            best_tree, _ = c.population.best_solution()
            y_pred = best_tree.predict_batch(X_train)
            accuracy_in = np.mean([(1 if y_pred[i] == y_train[i] else 0) for i in range(len(X_train))])
            y_pred = best_tree.predict_batch(X_test)
            accuracy_test = np.mean([(1 if y_pred[i] == y_test[i] else 0) for i in range(len(X_test))])

            histories[-1].append((elapsed_time, best_tree, (accuracy_in, accuracy_test)))
            save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)
        
        print(f"Saved to '{output_path_summ}'.")
        print(f"Saved to '{output_path_full}'.")