import pdb
import pandas as pd
import numpy as np

from rich import print
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
import VectorTree as vt
from SubstrateReal import SubstrateReal
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs, make_moons

DEparams = {"F":0.7, "Cr":0.8}

substrates_real = [
    # SubstrateReal("1point"),
    # SubstrateReal("2point"),
    # SubstrateReal("Multipoint"),
    # SubstrateReal("BLXalpha", {"Cr": 0.5}),
    # SubstrateReal("SBX", {"Cr": 0.5}),
    # SubstrateReal("Multicross", {"N": 5}),
    # SubstrateReal("Perm", {"Cr": 0.5}),
    SubstrateReal("MutRand", {"method": "Gauss", "F":0.01, "Cr": 0.01}),
    SubstrateReal("Gauss", {"F":0.001}),
    SubstrateReal("Laplace", {"F":0.001}),
    SubstrateReal("Cauchy", {"F":0.002}),
    SubstrateReal("DE/rand/1", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/best/1", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/rand/2", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/best/2", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/current-to-best/1", {"F":0.7, "Cr": 0.8}),
    SubstrateReal("DE/current-to-pbest/1", {"F":0.7, "Cr": 0.8}),
    # SubstrateReal("LSHADE", {"F":0.7, "Cr": 0.8}),
    # SubstrateReal("SA", {"F":0.01, "temp_ch": 20, "iter": 10}),
    # SubstrateReal("HS", {"F":0.01, "Cr":0.3, "Par":0.1}),
    # SubstrateReal("Replace", {"method":"Gauss", "F":0.1}),
    # SubstrateReal("Dummy", {"F": 100})
]

params = {
    "popSize": 100,
    "rho": 0.8,
    "Fb": 0.98,
    "Fd": 0.2,
    "Pd": 0.1,
    "k": 4,
    "K": 20,
    "group_subs": True,

    "stop_cond": "neval",
    "time_limit": 4000.0,
    "Ngen": 3500,
    "Neval": 1e5,
    "fit_target": 1000,

    "verbose": True,
    "v_timer": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 100,
    "prob_amp": 0.01
}

# Evolve the multivariate weights and infer leaf classes based on the training data
if __name__ == "__main__":
    depth = 5

    # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)

    # X, y = make_blobs(n_samples=1000, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    # y = np.array([y_i % 2 for y_i in y])

    # X, y = make_moons(n_samples=1000)

    df = pd.read_csv("achived/spiral.csv")
    X, y = df.iloc[:,:2], df.iloc[:,2]
    X, y = np.array(X), np.array(y)

    n_attributes = 2
    n_classes = 2

    y = np.array([y_i % 2 for y_i in y])

    class SupervisedObjectiveFunc(AbsObjetiveFunc):
        def __init__(self, size, opt="max"):
            self.size = size
            super().__init__(self.size, opt)

        def objetive(self, solution):
            W = solution.reshape((2**depth - 1, n_attributes + 1))
            accuracy, _ = vt.get_accuracy(X, y, W)
            
            # return accuracy

            penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(W)])
            alpha = 2

            # b = np.abs(W)
            # b[np.arange(len(b)), np.argmax(b[:, 1:], axis=1) + 1] = 0
            # penalty = np.sum(np.abs(W) - b)
            
            # alpha = 2

            return accuracy - alpha * penalty
        
        def random_solution(self):
            return vt.generate_random_weights(n_attributes, depth)
        
        def check_bounds(self, solution):
            return solution

    sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
    c = CRO_SL(SupervisedObjectiveFunc(sol_size), substrates_real, params)
    _, fit = c.optimize()

    final_weights, _ = c.population.best_solution()
    W = final_weights.reshape((2**depth - 1, n_attributes + 1))
    c.display_report()
    
    accuracy, labels = vt.get_accuracy(X, y, W)
    print(f"Accuracy: {accuracy}")
    plot_decision_surface(X, y, W, labels)

    print(vt.weights2treestr(W, labels))

    b = np.copy(W)
    b2 = np.abs(W)
    b[np.arange(len(b)), np.argmax(b2[:,1:], axis=1) + 1] = 0
    b[:, 0] = 0
    univ_W = W - b
    
    accuracy, labels = vt.get_accuracy(X, y, univ_W)
    print(f"Accuracy after univariating: {accuracy}")
    plot_decision_surface(X, y, univ_W, labels)

    print(vt.weights2treestr(univ_W, labels))
