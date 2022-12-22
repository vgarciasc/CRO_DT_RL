import pdb
import numpy as np

import time

from rich import print
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
from SoftTree import SoftTree, SoftTreeSigmoid
from SubstrateReal import SubstrateReal
from plotter import plot_decision_surface, plot_decision_surface_model
from sklearn.datasets import make_blobs, make_moons

DEparams = {"F":0.7, "Pr":0.8}
substrates_real = [
    SubstrateReal("SBX", {"F":0.8}),
    SubstrateReal("Perm", {"F":0.6}),
    SubstrateReal("1point"),
    SubstrateReal("2point"),
    SubstrateReal("Multipoint"),
    # SubstrateReal("Multicross", {"n_ind": 3}),
    SubstrateReal("BLXalpha", {"F":0.8}),
    SubstrateReal("Replace", {"method": "Uniform", "F":1/3}),
    SubstrateReal("DE/best/1", DEparams),
    SubstrateReal("DE/rand/1", DEparams),
    SubstrateReal("DE/best/2", DEparams),
    SubstrateReal("DE/rand/2", DEparams),
    SubstrateReal("DE/current-to-best/1", DEparams),
    SubstrateReal("DE/current-to-rand/1", DEparams),
    SubstrateReal("LSHADE", {"F":0.5, "Pr":0.5}),
    #SubstrateReal("DE/current-to-rand/1", DEparams),
    #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
    #SubstrateReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
    #SubstrateReal("Cauchy", {"F":0.005}),
    #SubstrateReal("Cauchy", {"F":0.005}),
    #SubstrateReal("Gauss", {"F":0.5}),
    #SubstrateReal("Gauss", {"F":0.05}),
    #SubstrateReal("Gauss", {"F":0.001}),
    #SubstrateReal("DE/rand/1", DEparams),
    #SubstrateReal("DE/best/2", DEparams),
    #SubstrateReal("DE/current-to-pbest/1", DEparams),
    #SubstrateReal("DE/current-to-best/1", DEparams),
    #SubstrateReal("DE/current-to-rand/1", DEparams)
]

params = {
    "ReefSize": 100,
    "rho": 0.7,
    "Fb": 0.98,
    "Fd": 0.8,
    "Pd": 0.2,
    "k": 7,
    "K": 3,
    "group_subs": True,

    "stop_cond": "ngen",
    "time_limit": 40.0,
    "Ngen": 100,
    "Neval": 3e5,
    "fit_target": 1000,

    "verbose": True,
    "v_timer": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "med",
    "dyn_steps": 75,
    "prob_amp": 0.05
}

# Evolve the multivariate weights and infer leaf classes based on the training data
if __name__ == "__main__":
    DEPTH = 5

    tree = SoftTree(num_attributes=2, num_classes=2)
    tree.randomize(depth=DEPTH)
    print(tree)

    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    # X, y = make_blobs(n_samples=1000, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    # X, y = make_moons(n_samples=1000)
    y = np.array([y_i % 2 for y_i in y])

    def dt_matrix_fit(weights):
        tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

        tree.update_leaves_by_dataset(X, y)
        y_pred = tree.predict_batch(X)
        accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])

        return accuracy

    alpha = 1
    def dt_matrix_fit_with_penalty(weights):
        accuracy = dt_matrix_fit(weights)
        penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)])

        return accuracy - alpha * penalty

    class SupervisedObjectiveFunc(AbsObjetiveFunc):
        def __init__(self, size, opt="max"):
            self.size = size
            super().__init__(self.size, opt)

        def objetive(self, solution):
            num_attributes = 2
            num_classes = 2

            tree = SoftTree(num_attributes=num_attributes, num_classes=num_classes,
                weights=solution.reshape((len(solution) // (num_attributes + 1), (num_attributes + 1))),
                labels=np.ones((len(solution) // (num_attributes + 1) + 1, num_classes)))
            tree.update_leaves_by_dataset(X, y)
            y_pred = tree.predict_batch(X)
            accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])

            penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(tree.weights)])
            alpha = 1

            return accuracy - alpha * penalty
        
        def random_solution(self):
            tree.randomize(depth=DEPTH)
            return tree.weights.flatten()
        
        def check_bounds(self, solution):
            return solution

    sol_size = len(tree.weights.flatten())
    c = CRO_SL(SupervisedObjectiveFunc(sol_size), substrates_real, params)
    _, fit = c.optimize()

    x, _ = c.population.best_solution()
    
    tree.weights = x.reshape((tree.num_nodes, tree.num_attributes + 1))
    tree.update_leaves_by_dataset(X, y)
    print(tree)
    print(f"Accuracy: {dt_matrix_fit(tree.weights)}")

    plot_decision_surface_model(X, y, tree)

    pdb.set_trace()