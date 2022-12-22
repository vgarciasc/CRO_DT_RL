import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sup_configs import get_config, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

import argparse

def fit_tree(data_config, seed, desired_depth=2, should_norm=False):
    X, y = load_dataset(data_config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=seed, stratify=y_test)

    if should_norm:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    
    valid_trees = []
    for alpha in np.linspace(0, 0.5, 1000):
        dt = DecisionTreeClassifier(ccp_alpha=alpha)
        dt.fit(X_train, y_train)

        if dt.get_depth() <= desired_depth:
            valid_trees.append(dt)
    
    best_acc_val = 0
    best_dt = None
    for dt in valid_trees:
        acc_val = dt.score(X_val, y_val)
        if acc_val > best_acc_val or best_dt is None:
            best_acc_val = acc_val
            best_dt = dt

    acc_in = best_dt.score(X_train, y_train)
    acc_out = best_dt.score(X_test, y_test)
    
    return best_dt, alpha, acc_in, acc_out

def get_cart_as_W(data_config, model, desired_depth):
    n_attributes = data_config["n_attributes"]

    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    features = model.tree_.feature
    thresholds = model.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    
    stack = [(0, 0)]
    W = []

    while len(stack) > 0:
        node_id, depth = stack.pop(0)
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            weights = np.zeros(n_attributes + 1)
            weights[features[node_id]] = 1
            weights[0] = - thresholds[node_id]
            W.append(weights)

            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            for d in range(2**(desired_depth - depth) - 1):
                weights = np.zeros(n_attributes + 1)
                weights[1] = 1
                weights[0] = 1
                W.append(weights)

            is_leaves[node_id] = True

    return np.array(W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CART Initializer')
    parser.add_argument('-i','--dataset',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('-o','--output',help="What file to write?", required=True, type=str)
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    dataset_list = ["breast_cancer", "car", "banknote", "balance", "acute-1", "acute-2", "transfusion", "climate", "sonar", "optical", "drybean", "avila", "wine-red", "wine-white"]
    if args['dataset'] == 'all':
        data_configs = [get_config(d) for d in dataset_list]
    elif args['dataset'].endswith("onwards"):
        dataset_start = dataset_list.index(args['dataset'][:-len("_onwards")])
        data_configs = [get_config(d) for d in dataset_list[dataset_start:]]
    else:
        data_configs = [get_config(args['dataset'])]
    
    initial_pop = []
    for data_config in data_configs:    
        acc_ins = []
        acc_outs = []
        weights = []
        
        print(f"Dataset: {data_config['name']}")
        for sample in range(args["simulations"]):
            dt, alpha, acc_in, acc_out = fit_tree(data_config, sample, 
                args["depth"], should_norm=args["should_normalize_dataset"])
        
            if args["verbose"]:
                print(f"Sample {sample} training: {acc_in}")
                print(f"Sample {sample} test: {acc_out}")
        
            w = get_cart_as_W(data_config, dt, args["depth"])
            weights.append(w)
            acc_ins.append(acc_in)
            acc_outs.append(acc_out)
                
        initial_pop.append((data_config["code"], weights))

        print(f"Dataset: {data_config['name']}")
        print(f"Accuracy in-sample: {'{:.3f}'.format(np.mean(acc_ins))} ± {'{:.3f}'.format(np.std(acc_ins))}")
        print(f"Accuracy out-of-sample: {'{:.3f}'.format(np.mean(acc_outs))} ± {'{:.3f}'.format(np.std(acc_outs))}")
        print()

    with open(args["output"], "wb") as f:
        pickle.dump(initial_pop, f)