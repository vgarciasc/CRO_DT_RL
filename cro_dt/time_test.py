import time
import numpy as np

from rich.console import Console
from rich import print

import cro_dt.VectorTree as vt
from cro_dt.sup_configs import get_config, load_dataset
console = Console()

if __name__ == "__main__":
    # X, y = load_dataset("breast_cancer")
    # X = np.array([[-3, -2], [-4, -2], [3, 3]])
    # y = np.array([1, 1, 0])
    W = np.array([[2, 3, -4], [5, -5, -4], [6, -2, 3]])
    X = np.array([[-3, -2], [4, -2], [3, 3], [-3, 1], [-2, -2], [-4, -4]])
    y = np.array([1, 1, 0, 1, 0, 0])
    # W = np.array([[2, 3, -4], [5, -5, -4], [4, 0, 2], [2, 0, 1], [6, -2, 3], [1, 3, 2], [3, 2, 4]])
    depth = int(np.log2(len(W) + 1))
    n_classes = np.max(y) + 1

    M = vt.create_mask_dx(depth)
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.tile(y, (len(W) + 1, 1))

    start_time = time.time()
    for _ in range(10000):
        acc1, labels1 = vt.dt_matrix_fit(X, y, W, -M, X_, Y_ + vt.MAGIC_NUMBER)
    end_time = time.time()

    console.rule("[red]Original vector tree implementation[/red]")
    print(f"Labels: {labels1}")
    print(f"Accuracy: {acc1}")
    print(f"Elapsed time: {'{:.3f}'.format(end_time - start_time)} seconds")

    start_time = time.time()
    for _ in range(10000):
        acc2, labels2 = vt.dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()

    console.rule("[red]New vector tree implementation[/red]")
    print(f"Labels: {labels2}")
    print(f"Accuracy: {acc2}")
    print(f"Elapsed time: {'{:.3f}'.format(end_time - start_time)} seconds")

    start_time = time.time()
    for _ in range(10000):
        acc2, labels2 = vt.dt_matrix_fit_dx2(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()

    console.rule("[red]Vector tree implementation but using numpy methods instead of pure matrix stuff[/red]")
    print(f"Labels: {labels2}")
    print(f"Accuracy: {acc2}")
    print(f"Elapsed time: {'{:.3f}'.format(end_time - start_time)} seconds")
    
    # solution.update_leaves_by_dataset(X_train, y_train)
    # y_pred = solution.predict_batch(X_train)
    # accuracy = np.mean([(1 if y_pred[i] == y_train[i] else 0) for i in range(len(X_train))])
    # return accuracy