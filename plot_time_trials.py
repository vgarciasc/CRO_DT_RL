import matplotlib.pyplot as plt
# from cycler import cycler
import numpy as np

if __name__ == "__main__":
    # plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    datasets = ["N=100, C=3, P=2", "N=1000, C=3, P=2", "N=1000, C=3, P=10",
        "N=1000, C=10, P=10", "N=10000, C=3, P=2", "N=10000, C=3, P=10",
        "N=100000, C=10, P=10"]
    models = ["Matrix $d=2$"]

    matrix_d2_values = [1.490, 1.213, 0.582, 0.584, 1.452, 1.456, 16.956]
    tree_d2_values = [13.644, 129.058, 128.424, 114.120, 1127.885, 1128.719, 0]

    matrix_d3_values = []
    matrix_d4_values = []

    X = np.arange(len(datasets))

    plt.bar(X - 0.125, matrix_d2_values, label="Matrix $d=2$", width=0.25)
    plt.bar(X + 0.125, tree_d2_values, label="Tree $d=2$", width=0.25)
    # plt.bar(X + 0.50, matrix_d2_values, width = 0.25)

    plt.legend()
    plt.ylabel("Average time for execution (s)")
    plt.xticks(range(len(datasets)), datasets, rotation=20)
    plt.yscale("log")
    plt.subplots_adjust(top=0.978, bottom=0.143, left=0.051, right=0.989, hspace=0.2, wspace=0.2)
    plt.show()
