import argparse
from datetime import datetime
import json

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, \
    train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sup_configs import get_config, load_dataset
import pdb
import time
import numpy as np

from rich import print
from utils import printv

def save_svm_histories_to_file(configs, histories, output_path, prefix=""):
    string = prefix

    for config, history in zip(configs, histories):
        elapsed_times, acc_ins, acc_outs = zip(*history)

        string += "--------------------------------------------------\n\n"
        string += f"DATASET: {config['name']}\n"
        string += f"Average in-sample SVM accuracy: {'{:.3f}'.format(np.mean(acc_ins))} ± {'{:.3f}'.format(np.std(acc_ins))}\n"
        string += f"Average test SVM accuracy: {'{:.3f}'.format(np.mean(acc_outs))} ± {'{:.3f}'.format(np.std(acc_outs))}\n"
        string += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(string)

def get_best_svm(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(gamma='auto', random_state=0, class_weight="balanced"))])
    parameter_space = {
        'svm__C': [2**p for p in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17]]}

    grid = GridSearchCV(pipe, parameter_space, cv=20, n_jobs=8, scoring="accuracy")
    grid.fit(X, y)

    print(f"Mean test score: {grid.cv_results_['mean_test_score']}")
    print(f"Rank: {grid.cv_results_['rank_test_score']}")
    
    return grid.best_estimator_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRO-SL for Supervised Tree Induction')
    parser.add_argument('-i','--dataset',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c','--cro_config',help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations?", required=True, type=int)
    parser.add_argument('-d','--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--alpha',help="How to penalize tree multivariateness?", required=True, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    depth = args["depth"]
    alpha = args["alpha"]
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)

    command_line = str(args)
    command_line += "\n\npython svm_baseline.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = f"results/log_{curr_time}_SVM.txt"

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

            start_time = time.time()
            
            best_svm = get_best_svm(X_train, y_train)
            best_svm.fit(X_train, y_train)
            acc_in = best_svm.score(X_train, y_train)
            acc_out = best_svm.score(X_test, y_test)

            end_time = time.time()

            histories[-1].append((end_time - start_time, acc_in, acc_out))
            save_svm_histories_to_file(data_configs, histories, output_path, command_line)
        
        print(f"Saved to '{output_path}'.")