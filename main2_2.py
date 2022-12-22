import pdb
import pandas as pd
import numpy as np

from rich import print
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
import VectorTree as vt
import reevaluator as rv
from SubstrateReal import SubstrateReal
from plotter import plot_decision_surface
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DEparams = {"F":0.7, "Cr":0.8}

substrates_real = [
    # SubstrateReal("1point"),
    # SubstrateReal("2point"),
    # SubstrateReal("Multipoint"),
    # SubstrateReal("BLXalpha", {"Cr": 0.5}),
    # SubstrateReal("SBX", {"Cr": 0.5}),
    # SubstrateReal("Multicross", {"N": 5}),
    # SubstrateReal("Perm", {"Cr": 0.5}),
    # SubstrateReal("MutRand", {"method": "Gauss", "F":0.01, "Cr": 0.01}),
    # SubstrateReal("Gauss", {"F":0.001}),
    # SubstrateReal("Laplace", {"F":0.001}),
    # SubstrateReal("Cauchy", {"F":0.002}),
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
    "Neval": 1e4,
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
    depth = 2

    # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)

    X, y = make_blobs(n_samples=1000, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    y = np.array([y_i % 2 for y_i in y])

    # X, y = make_moons(n_samples=1000)

    # df = pd.read_csv("achived/spiral.csv")
    # X, y = df.iloc[:,:2], df.iloc[:,2]
    # X, y = np.array(X), np.array(y)
    # y = np.array([y_i % 2 for y_i in y])

    n_attributes = 2
    n_classes = 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42, stratify=y)
    X_train_unsc, X_test_unsc, _, _ = train_test_split(X, y, test_size=0.01, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    class SupervisedObjectiveFunc(AbsObjetiveFunc):
        def __init__(self, size, opt="max"):
            self.size = size
            super().__init__(self.size, opt)

        def objetive(self, solution):
            W = solution.reshape((2**depth - 1, n_attributes + 1))
            accuracy, _ = vt.dt_matrix_fit(X_train, y_train, W)
            penalty = vt.get_penalty(W, alpha=1, should_normalize_rows=True, \
                should_apply_exp=False)

            return accuracy - penalty
        
        def random_solution(self):
            return vt.generate_random_weights(n_attributes, depth)
        
        def check_bounds(self, solution):
            return np.clip(solution.copy(), -1, 1)

    sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
    c = CRO_SL(SupervisedObjectiveFunc(sol_size), substrates_real, params)
    _, fit = c.optimize()

    multiv_W, _ = c.population.best_solution()
    multiv_W = multiv_W.reshape((2**depth - 1, n_attributes + 1))
    univ_W = vt.get_W_as_univariate(multiv_W)
    
    _, multiv_labels = vt.dt_matrix_fit(X_train, y_train, multiv_W)
    _, univ_labels = vt.dt_matrix_fit(X_train, y_train, univ_W)
    multiv_acc_in = vt.calc_accuracy(X_train, y_train, multiv_W, multiv_labels)
    univ_acc_in = vt.calc_accuracy(X_train, y_train, univ_W, univ_labels)
    multiv_acc_out = vt.calc_accuracy(X_test, y_test, multiv_W, multiv_labels)
    univ_acc_out = vt.calc_accuracy(X_test, y_test, univ_W, univ_labels)

    print(vt.weights2treestr(multiv_W, multiv_labels, None, False, scaler))
    print(vt.weights2treestr(univ_W, univ_labels, None, False, scaler))

    print(f"Multivariate accuracy in-sample: {multiv_acc_in}")
    print(f"Univariate accuracy in-sample: {univ_acc_in}")
    print(f"Multivariate accuracy out-of-sample: {multiv_acc_out}")
    print(f"Univariate accuracy out-of-sample: {univ_acc_out}")
    print()