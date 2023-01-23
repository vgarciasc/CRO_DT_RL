import matplotlib.pyplot as plt
import numpy as np
import pdb
from cycler import cycler


if __name__ == "__main__":
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    datasets = ["Artificial_1\n(N=100, C=3, P=2)",
        "Artificial_2\n(N=1000, C=3, P=2)",
        "Artificial_3\n(N=1000, C=3, P=10)",
        "Artificial_4\n(N=1000, C=10, P=10)",
        "Artificial_5\n(N=10000, C=3, P=2)",
        "Artificial_6\n(N=10000, C=3, P=10)",
        "Artificial_7\n(N=100000, C=10, P=10)"]
    
    tree_avgs = np.array([[1.84, 12.62, 12.66, 12.67, 120.53, 120.35, 1193.30],
        [2.47, 17.60, 17.67, 17.66, 168.99, 169.27, 1687.59],
        [3.15, 23.08, 23.19, 23.13, 218.01, 217.92, 2175.19],
        [4.04, 28.43, 28.58, 28.39, 268.41, 267.09, 2678.20],
        [5.28, 34.00, 34.33, 34.42, 317.50, 318.12, 3165.52],
        [7.36, 40.71, 41.13, 41.33, 369.63, 368.32, 3675.48],
        [11.65, 49.80, 50.45, 50.58, 427.75, 428.20, 4249.49],
        [19.72, 62.77, 64.01, 64.52, 490.83, 487.78, 4770.71]])

    tree_stds = np.array([[0.01, 0.10, 0.06, 0.06, 0.75, 0.73, 3.51],
        [0.01, 0.11, 0.10, 0.07, 1.13, 0.50, 6.03],
        [0.02, 0.07, 0.10, 0.07, 1.19, 0.90, 4.88],
        [0.02, 0.11, 0.06, 0.12, 1.02, 1.20, 11.35],
        [0.03, 0.06, 0.10, 0.11, 0.32, 1.06, 8.32],
        [0.01, 0.09, 0.07, 0.06, 0.62, 0.97, 6.52],
        [0.11, 0.13, 0.06, 0.13, 1.83, 0.51, 11.88],
        [0.06, 0.12, 0.25, 0.55, 1.44, 3.03, 9.95]])

    matrix_avgs = np.array([[0.46, 0.55, 0.56, 0.56, 1.41, 1.42, 13.26],
        [0.54, 0.73, 0.76, 0.75, 3.98, 4.07, 27.36],
        [0.77, 1.17, 1.22, 1.20, 10.35, 10.09, 107.63],
        [0.97, 3.13, 3.31, 3.32, 24.55, 24.53, 253.68],
        [1.60, 5.73, 5.82, 5.82, 51.18, 49.83, 521.91],
        [5.88, 16.59, 19.08, 18.44, 111.89, 113.05, 1416.77],
        [11.63, 41.73, 41.83, 39.35, 252.13, 235.69, 2838.02],
        [37.51, 130.53, 170.81, 127.57, 694.26, 703.08, 7001.62]])
    
    matrix_stds = np.array([[0.01, 0.00, 0.01, 0.00, 0.01, 0.01, 0.12],
        [0.00, 0.01, 0.01, 0.01, 0.19, 0.04, 0.17],
        [0.01, 0.01, 0.01, 0.01, 3.14, 2.60, 25.28],
        [0.00, 0.16, 0.05, 0.04, 2.06, 2.08, 59.35],
        [0.01, 0.08, 0.04, 0.10, 5.21, 5.08, 128.84],
        [0.10, 5.53, 7.25, 7.14, 24.86, 29.24, 9.86],
        [0.12, 12.94, 12.08, 12.21, 64.96, 62.58, 6.63],
        [3.00, 31.25, 69.44, 46.15, 115.14, 117.83, 121.27]])

    fig, axs = plt.subplots(2, 4, sharex=True, figsize=(12, 6))
    axs[-1, -1].axis('off')
    for i, dataset in enumerate(datasets):
        x = range(2, 10)
        ax = axs[i//4, i%4]
        ax.plot(x, tree_avgs[:,i], marker="*", label="Traditional tree encoding", color="red")
        ax.fill_between(x, tree_avgs[:,i] - tree_stds[:,i], tree_avgs[:,i] + tree_stds[:,i], color="red", alpha=0.2)
        ax.plot(x, matrix_avgs[:,i], marker="*", label="Proposed matrix encoding", color="blue")
        ax.fill_between(x, matrix_avgs[:,i] - matrix_stds[:,i], matrix_avgs[:,i] + matrix_stds[:,i], color="blue", alpha=0.2)
        ax.set_title(dataset)
        ax.set_yscale("log")
        
        if i >= 1*4 or i == 3:
            ax.set_xlabel("Depth")
    
        if i == 6:
            ax.legend(bbox_to_anchor=(1.2, 0.5), loc="center left")

    axs[0, 0].set_ylabel("Training time (seg)")
    axs[1, 0].set_ylabel("Training time (seg)")
    plt.subplots_adjust(top=0.917,
        bottom=0.137,
        left=0.066,
        right=0.986,
        hspace=0.276,
        wspace=0.244)
    plt.show()