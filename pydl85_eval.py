import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
from pydl85 import DL85Classifier

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
    
    dt = DL85Classifier(max_depth=desired_depth)
    dt.fit(X_train, y_train)

    acc_in = dt.score(X_train, y_train)
    acc_out = dt.score(X_test, y_test)
    
    return dt, acc_in, acc_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CART Initializer')
    parser.add_argument('-i','--dataset',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('-o','--output',help="What file to write?", required=True, type=str)
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

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
    
    initial_pop = []
    for data_config in data_configs:    
        acc_ins = []
        acc_outs = []
        trees = []
        
        print(f"Dataset: {data_config['name']}")
        for sample in range(args["simulations"]):
            dt, acc_in, acc_out = fit_tree(data_config, sample, 
                args["depth"], should_norm=args["should_normalize_dataset"])
        
            if args["verbose"]:
                print(f"Sample {sample} training: {acc_in}")
                print(f"Sample {sample} test: {acc_out}")
        
            trees.append(dt)
            acc_ins.append(acc_in)
            acc_outs.append(acc_out)
                
        initial_pop.append((data_config["code"], trees))

        print(f"Dataset: {data_config['name']}")
        print(f"Accuracy in-sample: {'{:.3f}'.format(np.mean(acc_ins))} ± {'{:.3f}'.format(np.std(acc_ins))}")
        print(f"Accuracy out-of-sample: {'{:.3f}'.format(np.mean(acc_outs))} ± {'{:.3f}'.format(np.std(acc_outs))}")
        print()

    pdb.set_trace()