import numpy as np
import VectorTree as vt
import sup_configs as configs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parse_str_univ(string, config):
    lines = [line.strip() for line in string.split("\n")]

    weights = []
    labels = []

    for line in lines:
        depth = line.rindex("- ") + 1
        content = line[depth:].strip()
        is_leaf = "<=" not in content

        if not is_leaf:
            attribute, threshold = content.split(" <= ")
            
            attribute = int(attribute[1:])
            threshold = float(threshold)

            w = np.zeros(config["n_attributes"] + 1)
            w[attribute] = 1
            w[0] = -threshold
            weights.append(w)
        if is_leaf:
            labels.append(int(content))
    
    return np.array(weights), np.array(labels)

if __name__ == "__main__":
    seed = 0

    config = configs.config_DB
    X, y = configs.load_dataset(config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=seed, stratify=y_test)
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # X = scaler.transform(X)

    # tree_str = "- x2 <= 0.13039289228618145\n--- 0\n--- 1"
    # tree_str = "- x7 <= -0.2504889480769634\n-- x2 <= 0.13039289228618145\n--- 2\n--- 4\n-- x4 <= -0.23575056716799736\n--- 2\n--- 4"
    tree_str = "- x3 <= 280.70419\n-- x13 <= 0.00682\n--- x15 <= 0.72592\n---- 5\n---- 0\n--- x2 <= 705.50052\n---- 6\n---- 6\n-- x12 <= 0.74161\n--- x4 <= 215.31133\n---- 4\n---- 3\n--- x2 <= 897.31650\n---- 5\n---- 3"
    W, labels = parse_str_univ(tree_str, config)
    # W = np.array([
    #     [-0.27391   ,  1.        ,  0.        ,  0.        ,  0.        ],
    #     [-7.56529999,  0.        ,  1.        ,  0.        ,  0.        ],
    #     [ 4.38604999,  0.        ,  0.        ,  1.        ,  0.        ]])
    # labels = np.array([1, 0, 1, 0])
    print(W)

    # accuracy, labels = vt.dt_matrix_fit(X_train, y_train, W)

    acc_total = vt.calc_accuracy(X, y, W, labels)
    acc_in = vt.calc_accuracy(X_train, y_train, W, labels)
    acc_out = vt.calc_accuracy(X_test, y_test, W, labels)

    # print(vt.weights2treestr(W, labels, None, False, scaler))

    print(f"Univariate accuracy total: {acc_total}")
    print(f"Univariate accuracy in-sample: {acc_in}")
    print(f"Univariate accuracy out-of-sample: {acc_out}")

    # print(tree_str)
    print()
    print(vt.weights2treestr(W, labels, config))
    print()
    print(vt.calc_accuracy(X, y, W, labels))
    print()
    import pdb
    pdb.set_trace()
