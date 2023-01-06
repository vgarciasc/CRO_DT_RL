from sklearn.datasets import make_classification, make_multilabel_classification
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdb

if __name__ == "__main__":
    n_classes = 10
    n_features = 10
    N = 100000
    
    filename = f"datasets/artificial/scikit_N-{N}_C-{n_classes}_P-{n_features}.csv"

    X, y = make_classification(n_samples=N, n_features=n_features, 
        n_informative=n_features, n_redundant=0, 
        n_classes=n_classes, n_clusters_per_class=1)
    y = np.array(np.reshape(y, (N, 1)), dtype=np.int32)
    df = pd.DataFrame(data=np.hstack((X, y)))
    df.to_csv(filename, header=False, index=False)
    print(f"Saved to '{filename}'.")

    # if should_display:
    #     for c in range(n_classes):
    #         class_c = np.array([x for (i, x) in enumerate(X) if y[i] == c])
    #         plt.scatter(class_c[:,0], class_c[:,1], label=f"Class {c}")
    #     plt.legend()
    #     plt.show()
