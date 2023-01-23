import matplotlib.pyplot as plt
# from cycler import cycler
import numpy as np

if __name__ == "__main__":
    # plt.rcParams['axes.prop_cycle'] = cycler(color=["#d16f6f", "#d43b3b", "#f51616", "#837cfc", "#433ae0", "#1409e0"])

    datasets = ["N=100, C=3, P=2", "N=1000, C=3, P=2", "N=1000, C=3, P=10",
        "N=1000, C=10, P=10", "N=10000, C=3, P=2", "N=10000, C=3, P=10",
        "N=100000, C=10, P=10"]
    models = ["Matrix $d=2$"]

    matrix_d2_values = [1.490, 1.213, 0.582, 0.584, 1.452, 1.456, 16.956]
    tree_d2_values = [12.703, 115.863, 116.231, 116.907, 1141.903, 1145.557, 0]

    matrix_d3_values = [0.561, 0.757, 0.781, 0.776, 5.317, 4.533, 27.519]
    tree_d3_values = [14.250, 129.680, 131.233, 131.216, 1289.945, 1292.297, 0]

    matrix_d4_values = [0.683, 1.078, 1.123, 1.130, 10.684, 9.451, 105.820]
    tree_d4_values = [15.933, 140.211, 143.193, 143.423, 1386.285, 1387.350, 0]

    X = np.arange(len(datasets))

    plt.bar(X - 0.3, matrix_d2_values, label="Matrix $d=2$", width=0.1, edgecolor="black")
    plt.bar(X - 0.2, matrix_d3_values, label="Matrix $d=3$", width=0.1, edgecolor="black")
    plt.bar(X - 0.1, matrix_d4_values, label="Matrix $d=4$", width=0.1, edgecolor="black")
    plt.bar(X, tree_d2_values, label="Tree $d=2$",  width=0.1, edgecolor="black")
    plt.bar(X + 0.1, tree_d3_values, label="Tree $d=3$", width=0.1, edgecolor="black")
    plt.bar(X + 0.2, tree_d4_values, label="Tree $d=4$", width=0.1, edgecolor="black")
    # plt.bar(X + 0.50, matrix_d2_values, width = 0.25)

    plt.legend()
    plt.ylabel("Average time for execution (s)")
    plt.xticks(range(len(datasets)), datasets, rotation=20)
    plt.yscale("log")
    plt.subplots_adjust(top=0.978, bottom=0.143, left=0.051, right=0.989, hspace=0.2, wspace=0.2)
    plt.show()
