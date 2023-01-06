import math
import argparse
import copy
from datetime import datetime
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
from CoralPopulation import Coral
from SubstrateReal import SubstrateReal
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
        print(f"Different CART solutions found: {len(cart_pop)}")

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

def get_W_from_solution(solution, depth, n_attributes, args):
    W = solution.reshape((2**depth - 1, n_attributes + 1))

    if args["should_use_univariate_accuracy"]:
        W = vt.get_W_as_univariate(W)

    if args["should_use_threshold"]:
        W[:,1:][abs(W[:,1:]) < args["threshold"]] = 0
        W[:,0][W[:,0] == 0] += 0.01
    
    return W

def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix + "\n"
    string_full = prefix + "\n"

    for config, history in zip(configs, histories):
        # elapsed_times, multiv_info, univ_info = zip(*history)
        elapsed_times, scalers, multiv_info, univ_info = zip(*history)
        multiv_acc_in, multiv_acc_test, multiv_W, multiv_labels = zip(*multiv_info)
        univ_acc_in, univ_acc_test, univ_W, univ_labels = zip(*univ_info)

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"{len(elapsed_times)} simulations executed.\n"
        string_summ += f"Average in-sample multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_in))} ± {'{:.3f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average in-sample univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_in))} ± {'{:.3f}'.format(np.std(univ_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_test))} ± {'{:.3f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += f"Average test univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_test))} ± {'{:.3f}'.format(np.std(univ_acc_test))}\n"
        string_summ += "\n"
        string_summ += f"Best test multivariate accuracy: {'{:.3f}'.format(multiv_acc_test[np.argmax(multiv_acc_test)])}\n"
        string_summ += f"Best test univariate accuracy: {'{:.3f}'.format(univ_acc_test[np.argmax(univ_acc_test)])}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += "--------------------------------------------------\n\n"
        string_full += f"DATASET: {config['name']}\n"
        
        for (elapsed_time, \
            (scaler), \
            (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
            (univ_acc_in, univ_acc_test, univ_labels, univ_W)) in history:

            string_full += f"In-sample:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_in}" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_in}" + "\n"
            string_full += f"Test:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_test}" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_test}" + "\n"
            # if scaler is not None:
            #     string_full += f"Scaler: (mean, {scaler.mean_})\n"
            #     string_full += f"        (var,  {scaler.var_})\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            string_full += "\n"
            string_full += "Multivariate tree:\n" + vt.weights2treestr(multiv_W, multiv_labels, config, use_attribute_names=False, scaler=scaler)
            string_full += f"\n"
            string_full += f"Multivariate labels: {multiv_labels}\n"
            string_full += str(multiv_W)
            string_full += f"\n\n"
            string_full += "Univariate tree:\n" + vt.weights2treestr(univ_W, univ_labels, config, use_attribute_names=False, scaler=scaler)
            string_full += f"\n"
            string_full += f"Univariate labels: {univ_labels}\n"
            string_full += str(univ_W)
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
    parser.add_argument('--should_get_best_from_validation', help='Should get best solution from validation set?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_apply_exponential', help='Should apply exponential penalty?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_use_threshold', help='Should ignore weights under a certain threshold?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--threshold', help="Under which threshold should weights be ignored?", required=False, default=0.05, type=float)
    parser.add_argument('--should_use_univariate_accuracy', help='Should use univariate tree\'s accuracy when measuring fitness?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--start_from', help='Should start from where?', required=False, default=0, type=int)
    parser.add_argument('--evaluation_scheme', help='Which evaluation scheme to use?', required=False, default="dx", type=str)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="log", type=str)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    depth = args["depth"]
    alpha = args["alpha"]
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)
    popsize = cro_configs["general"]["popSize"]

    command_line = str(args)
    command_line += "\n\npython main3.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path_summ = f"results/{args['output_prefix']}_{curr_time}_summary.txt"
    output_path_full = f"results/{args['output_prefix']}_{curr_time}_full.txt"
    
    normal_dataset_list = ["breast_cancer", "car", "banknote", "balance", "acute-1", "acute-2", "transfusion", "climate", "sonar", "optical", "drybean", "avila", "wine-red", "wine-white"]
    artificial_dataset_list = ["artificial_100_3_2", "artificial_1000_3_2", "artificial_1000_3_10", "artificial_1000_10_10", "artificial_10000_3_10", "artificial_10000_3_10", "artificial_100000_10_10"]

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

    if args["evaluation_scheme"] == "numba":
        fitness_evaluation = vt.dt_matrix_fit_dx_numba
    elif args["evaluation_scheme"] == "dx":
        fitness_evaluation = vt.dt_matrix_fit_dx
    elif args["evaluation_scheme"] == "dx2":
        fitness_evaluation = vt.dt_matrix_fit_dx
    elif args["evaluation_scheme"] == "old":
        fitness_evaluation = vt.dt_matrix_fit

    histories = []
    for data_config in data_configs:
        X, y = load_dataset(data_config)
        mask = vt.create_mask(depth)

        n_attributes = data_config["n_attributes"]
        n_classes = data_config["n_classes"]
        max_penalty = (n_attributes - 1) * (2 ** depth - 1)

        histories.append([])
        
        start_idx = args['start_from'] if data_config == data_configs[0] else 0
        simulations = range(args["simulations"])[start_idx:]

        for simulation in simulations:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation, stratify=y)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation, stratify=y_test)
            
            if args["should_normalize_dataset"]:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                # X_val = scaler.transform(X_val)
            else:
                scaler = None
            
            # X_prepared = np.vstack((np.ones(len(X_train)).T, X_train.T)).T
            # y_prepared = np.tile(y_train + vt.MAGIC_NUMBER, (2 ** depth, 1))
            
            M = vt.create_mask_dx(depth)
            X_ = np.vstack((np.ones(len(X_train)).T, X_train.T)).T
            Y_ = np.tile(y_train, (2**depth, 1))

            print(f"Iteration #{simulation}:")

            class SupervisedObjectiveFunc(AbsObjetiveFunc):
                def __init__(self, size, opt="max"):
                    self.size = size
                    super().__init__(self.size, opt)

                def objetive(self, solution):
                    W = get_W_from_solution(solution, depth, n_attributes, args)
                    accuracy, _ = fitness_evaluation(X_train, y_train, W, depth, n_classes, X_, Y_, M)

                    if args["should_use_univariate_accuracy"]:
                        return accuracy
                    else:
                        penalty = vt.get_penalty(W, max_penalty, alpha=args["alpha"], 
                            should_normalize_rows=args["should_normalize_rows"], \
                            should_normalize_penalty=args["should_normalize_penalty"], \
                            should_apply_exp=args["should_apply_exponential"])
                        return accuracy - penalty
                
                def random_solution(self):
                    return vt.generate_random_weights(n_attributes, depth)
                
                def check_bounds(self, solution):
                    return np.clip(solution.copy(), -1, 1)

            sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
            objfunc = SupervisedObjectiveFunc(sol_size)

            initial_pop = get_initial_pop(data_config, popsize, X_train, y_train,
                args["should_cart_init"], args["depth"], args["initial_pop"], objfunc)
            
            c = CRO_SL(objfunc, get_substrates_real(cro_configs), cro_configs["general"])
            if initial_pop is not None:
                c.population.population = []
                for tree in initial_pop:
                    coral = Coral(tree.flatten(), objfunc=objfunc)
                    coral.get_fitness()
                    c.population.population.append(coral)

            print(f"Average accuracy in CART seeding: {np.mean([f.fitness for f in c.population.population])}")
            print(f"Best accuracy in CART seeding: {np.max([f.fitness for f in c.population.population])}")
            pdb.set_trace()

            start_time = time.time()
            _, fit = c.optimize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            # c.display_report()

            if args["should_get_best_from_validation"]:
                # GET BEST SOLUTION BASED ON VALIDATION SET
                final_corals = [coral for coral in c.population.population]
                final_corals.sort(key = lambda x : x.fitness, reverse=True)
                aux = []
                fitnesses = []
                for coral in final_corals:
                    if coral.fitness in fitnesses:
                        continue
                    aux.append(coral)
                    fitnesses.append(coral.fitness)
                final_corals = aux
                final_corals = final_corals[:int(popsize*0.1)] if len(final_corals) > popsize*0.1 else final_corals 
                for i, coral in enumerate(final_corals[:25]):
                    print(f"Coral #{i+1}: (train: {coral.fitness})")

                X_val_ = np.vstack((np.ones(len(X_val)).T, X_val.T)).T
                Y_val_ = np.tile(y_val, (2**depth, 1))
                for coral in final_corals:
                    W = get_W_from_solution(coral.solution, depth, n_attributes, args)
                    
                    accuracy, _ = vt.dt_matrix_fit_dx(X_val, y_val, W, depth, n_classes, X_val_, Y_val_, M)
                    
                    if args["should_use_univariate_accuracy"]:
                        coral.val_fitness = accuracy
                    else:
                        penalty = vt.get_penalty(W, max_penalty, alpha=args["alpha"], 
                            should_normalize_rows=args["should_normalize_rows"], \
                            should_normalize_penalty=args["should_normalize_penalty"], \
                            should_apply_exp=args["should_apply_exponential"])
                        coral.val_fitness = accuracy - penalty

                final_corals.sort(key = lambda x : x.val_fitness, reverse=True)
                for i, coral in enumerate(final_corals):
                    print(f"Coral #{i+1}: (train: {coral.fitness}, val: {coral.val_fitness})")

                multiv_W = final_corals[0].solution
            else:
                multiv_W, _ = c.population.best_solution()

            # END REGION

            multiv_W = multiv_W.reshape((2**depth - 1, n_attributes + 1))
            if args["should_use_threshold"]:
                multiv_W[:,1:][abs(multiv_W[:,1:]) < args["threshold"]] = 0
                multiv_W[:,0][multiv_W[:,0] == 0] += 0.01
            univ_W = vt.get_W_as_univariate(multiv_W)
            if args["should_use_univariate_accuracy"]:
                multiv_W = vt.get_W_as_univariate(univ_W)
            
            _, multiv_labels = vt.dt_matrix_fit_dx(X_train, y_train, multiv_W, depth, n_classes)
            multiv_acc_in = vt.calc_accuracy(X_train, y_train, multiv_W, multiv_labels)
            multiv_acc_test = vt.calc_accuracy(X_test, y_test, multiv_W, multiv_labels)

            _, univ_labels = vt.dt_matrix_fit_dx(X_train, y_train, univ_W, depth, n_classes)
            univ_acc_in = vt.calc_accuracy(X_train, y_train, univ_W, univ_labels)
            univ_acc_test = vt.calc_accuracy(X_test, y_test, univ_W, univ_labels)

            histories[-1].append((elapsed_time, 
                (scaler), \
                (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
                (univ_acc_in, univ_acc_test, univ_labels, univ_W)))
            save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)
        
        print(f"Saved to '{output_path_summ}'.")
        print(f"Saved to '{output_path_full}'.")