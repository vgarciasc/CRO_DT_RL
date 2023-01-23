import pdb
import numpy as np
from functools import reduce
from numba import jit
import time
from rich import print

from cro_dt.sup_configs import load_dataset, config_CE

MAGIC_NUMBER = 42

def create_mask(depth=3):
    m = np.zeros((2**depth, 2**depth - 1))
    m[0] = np.concatenate([np.ones(depth), np.zeros(2**depth - 1 - depth)])

    for i in range(1, 2 ** depth):
        m[i] = m[i-1]

        last_pos = 0
        for j in range(2 ** depth - 1):
            if m[i-1][j] != 0:
                last_pos = j

        if m[i-1][last_pos] == 1:
            m[i][last_pos] = -1
            continue
            
        inverted_count = 0
        for j in range(2 ** depth - 2, -1, -1):
            if m[i-1][j] == -1:
                m[i][j] = 0

                if last_pos + 1 + inverted_count < 2 ** depth - 1:
                    m[i][last_pos + 1 + inverted_count] = 1
                    inverted_count += 1
            elif m[i-1][j] == 1:
                break
                
        for j in range(last_pos - 1, -1, -1):
            if m[i][j] == 1:
                m[i][j] = -1
                break
        
    return m

def create_mask_dx(depth=3):
    m = np.zeros((2**depth, 2**depth - 1))
    m[0] = np.concatenate([- np.ones(depth), np.zeros(2**depth - 1 - depth)])

    for i in range(1, 2 ** depth):
        m[i] = m[i-1]

        last_pos = 0
        for j in range(2 ** depth - 1):
            if m[i-1][j] != 0:
                last_pos = j

        if m[i-1][last_pos] == -1:
            m[i][last_pos] = 1
            continue
            
        inverted_count = 0
        for j in range(2 ** depth - 2, -1, -1):
            if m[i-1][j] == 1:
                m[i][j] = 0

                if last_pos + 1 + inverted_count < 2 ** depth - 1:
                    m[i][last_pos + 1 + inverted_count] = -1
                    inverted_count += 1
            elif m[i-1][j] == -1:
                break
                
        for j in range(last_pos - 1, -1, -1):
            if m[i][j] == -1:
                m[i][j] = 1
                break
        
    return m

def predict_batch(X, W, labels, add_1=False):
    if add_1:
        X = np.vstack((np.ones(len(X)).T, X.T)).T

    depth = int(np.log2(len(W) + 1))
    m = create_mask(depth=depth)

    K = m @ np.sign(- W @ X.T)
    L = np.clip(K - (np.max(K) - 1), 0, 1)
    leaves = np.argmax(L.T, axis=1)
    y = np.array([labels[l] for l in leaves])

    return y

def create_nodes_tree_mapper(depth):
    def process_node(node_idx, inners=[], leaves=[], curr_depth=0):
        if curr_depth == depth:
            return inners, leaves + [node_idx]
        else:
            inners += [node_idx]
            inners, leaves = process_node(node_idx + 1, inners, leaves, curr_depth+1)
            inners, leaves = process_node(node_idx + 2**(depth-curr_depth), inners, leaves, curr_depth+1)
            return inners, leaves

    inners, leaves = process_node(0)
    inners = {x:i for i,x in enumerate(inners)}
    leaves = {x:i for i,x in enumerate(leaves)}
    return inners, leaves

def dt_tree_fit(X, y, W, depth, n_classes, X_=None, Y_=None, M=None, default_label=0):
    n_leaves = len(W) + 1

    M_i, M_l = create_nodes_tree_mapper(depth) if M is None else M

    def get_leaf_idx(x, M_i, M_l):
        node_idx = 0
        curr_depth_2go = depth
        
        while curr_depth_2go != 0:
            if W[M_i[node_idx]] @ x <= 0:
                node_idx += 1
            else:
                node_idx = node_idx + 2**(curr_depth_2go)
            curr_depth_2go -= 1
        
        return M_l[node_idx]

    count = [np.zeros(n_classes) for _ in range(n_leaves)]
    
    for x_i, y_i in zip(X_, y):
        leaf = get_leaf_idx(x_i, M_i, M_l)
        count[leaf][y_i] += 1
    
    labels = np.zeros((n_leaves, n_classes))
    for leaf, samples in enumerate(count):
        labels[leaf][np.argmax(samples)] = 1

    accuracy = sum(np.max(np.array(count), axis=1)) / len(X)

    return accuracy, labels

def dt_matrix_fit(X, y, W, depth, n_classes, X_=None, Y_=None, M=None, default_label=0):
    num_leaves = len(W) + 1

    # Mask to detect the given leaf for each observation
    M = create_mask_dx(depth=depth) if M is None else M
    X = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y = np.tile(y, (num_leaves, 1)) if Y_ is None else Y_

    # Matrix composed of one-hot vectors defining the leaf index of each input
    K = - M @ np.sign(- W @ X.T)
    L = np.clip(K - (depth - 1), 0, 1)
    zero_count = len(X) - np.sum(L, axis=1)

    # Creating optimal label vector
    optimal_labels = np.ones(num_leaves) * (default_label)

    # Label matrix
    labeled_leaves = np.int_(L * Y)
    
    correct_inputs = 0
    for i, leaf in enumerate(labeled_leaves):
        # Array with the count for each label in given leaf
        bincount = np.bincount(leaf)
        bincount[0] -= zero_count[i]

        optimal_label = np.argmax(bincount)
        optimal_labels[i] = optimal_label

        correct_inputs += bincount[optimal_label]

    accuracy = correct_inputs / len(X)

    return accuracy, optimal_labels

clipper = lambda A, d : reduce(np.multiply, [(i - A) / (i - d) for i in range(-d, d)])
bincount = lambda A, i, k, N : reduce(np.multiply, [(j - A) / (j - i) for j in range(0, k) if j != i]) @ np.ones(N)

def dt_matrix_fit_dx(X, y, W, depth, n_classes, X_=None, Y_=None, M=None, untie=False):
    n_leaves = len(W) + 1
    N = len(X)

    M = create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_
    
    Z = np.sign(W @ X_.T)
    # Z = np.sign(np.sign(W @ X_.T) - 0.5) # Slightly more inefficient but guarantees that ties do not happen
    # Z_ = clipper(M @ Z, depth)
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_
    
    count_0s = N - np.sum(Z_, axis=1)
    R = np.int_(R)
    H = np.zeros((n_leaves, n_classes))
    labels = np.zeros(n_leaves)
    correct_preds = 0
    for l in range(n_leaves):
        bc = np.bincount(R[l], minlength=n_classes)
        bc[0] -= count_0s[l]

        most_popular_class = np.argmax(bc)
        labels[l] = most_popular_class
        correct_preds += bc[most_popular_class]

    accuracy = correct_preds / N
    return accuracy, labels

    # R = np.array(np.vstack((np.zeros(R.shape[0]), R.T)).T, dtype=np.int32)
    # H = np.array([np.bincount(r, minlength=n_classes) for r in R], dtype=np.float64)
    # H[:,0] -= (np.ones(n_leaves) * N - Z_ @ S) + 1 # removing false zeros

    H = np.stack([bincount(R, i, n_classes, N) for i in range(n_classes)])
    H[0] -= (np.ones(n_leaves) * N - np.sum(Z_, axis=1)) # removing false zeros

    labels = np.argmax(H, axis=0)
    accuracy = sum(np.max(H, axis=0)) / N

    return accuracy, labels

def dt_matrix_fit_dx2(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    n_leaves = len(W) + 1
    N = len(X)

    M = create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_
    S = np.ones(N)
    
    Z = np.sign(W @ X_.T)
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    H = np.stack([bincount(R, i, n_classes, N) for i in range(n_classes)])
    H[0] -= (np.ones(n_leaves) * N - Z_ @ S) # removing false zeros

    labels = np.argmax(H, axis=0)
    accuracy = sum(np.max(H, axis=0)) / N

    return accuracy, labels

def dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_=None, Y_=None, M=None, untie=False):
    n_leaves = len(W) + 1
    N = len(X)

    M = create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_
    S = np.ones(N)
    
    bc, Z_ = numba_subproc1(X_, Y_, M, S, W, depth, n_classes)
    H = np.stack(bc) @ np.ones(N)
    accuracy, labels = numba_subproc2(H, Z_, S, N, n_leaves)
    return accuracy, labels

@jit(nopython=True)
def numba_subproc1(X_, Y_, M, S, W, depth, n_classes):
    Z = np.sign(W @ X_.T)
    # Z = np.sign(np.sign(W @ X_.T) - 0.5) # Slightly more inefficient but guarantees that ties do not happen
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    bc = [np.ones_like(R) for _ in range(0, n_classes)]
    for i in range(0, n_classes):
        for j in range(0, n_classes):
            if j != i:
                bc[i] *= (j - R) / (j - i)
    
    return bc, Z_

@jit(nopython=True)
def numba_subproc2(H, Z_, S, N, n_leaves):
    H[0] -= (np.ones(n_leaves) * N - Z_ @ S) # removing false zeros
    
    labels = [np.argmax(r) for r in H.T]
    accuracy = sum([np.max(r) for r in H.T]) / N

    return accuracy, labels

def dt_matrix_fit_old(X, y, W, default_label=0):
    num_leaves = len(W) + 1
    depth = int(np.log2(len(W) + 1))

    X = np.vstack((np.ones(len(X)).T, X.T)).T
    Y = np.tile(y + MAGIC_NUMBER, (num_leaves, 1))

    # Mask to detect the given leaf for each observation
    m = create_mask(depth=depth)

    # Matrix composed of one-hot vectors defining the leaf index of each input
    K = m @ np.sign(- W @ X.T)
    L = np.clip(K - (np.max(K) - 1), 0, 1)

    # Creating optimal label vector
    optimal_labels = np.ones(num_leaves) * (default_label + MAGIC_NUMBER)

    # Label matrix
    labeled_leaves = np.int_(L * Y)
    
    correct_inputs = 0
    for i, leaf in enumerate(labeled_leaves):
        # Array with the count for each label in given leaf
        bincount = np.bincount(leaf)

        # Get the most common label, excluding 0
        if len(bincount) > 1:
            optimal_label = np.argmax(bincount[1:]) + 1
            optimal_labels[i] = optimal_label
        
        total_in_leaf = np.sum(bincount[1:])
        if total_in_leaf == 0:
            continue

        correct_inputs += bincount[optimal_label]

    accuracy = correct_inputs / len(X)
    optimal_labels = optimal_labels - MAGIC_NUMBER

    return accuracy, optimal_labels

def generate_random_weights(n_attributes, depth):
    return np.random.uniform(-1, 1, size=(2**depth - 1) * (n_attributes + 1))

def weights2treestr(weights, labels, data_config=None, use_attribute_names=False, scaler=None):
    if data_config is not None and use_attribute_names:
        attribute_names = data_config["attributes"]
    else:
        attribute_names = [f"x{i}" for i in range(len(weights[0]))]

    weights = weights.copy()
    labels = labels.copy()
    
    depth = int(np.log2(len(weights) + 1)) + 1
    stack = [(0, 1)]
    output = ""
    curr_leaf = 0
    curr_node = 0

    try:
        if scaler is not None:
            sigma = np.sqrt(np.array([1] + list(scaler.var_)))
            mu = np.array([0] + list(scaler.mean_))
            
            mask = np.zeros_like(weights)
            mask[:, 0] = 1

            offsets = ((weights / sigma) @ mu).reshape(len(weights), 1)
            weights = weights / sigma - np.multiply(mask, offsets)
    except:
        pdb.set_trace()

    while len(stack) > 0:
        node, curr_depth = stack.pop()
        output += "-" * curr_depth + " "

        if curr_depth == depth:
            output += str(int(labels[curr_leaf]))
            curr_leaf += 1
        else:
            non_zero_attributes = [i for i, w in enumerate(weights[curr_node][1:]) if np.abs(w) > 0.00000001]
            if len(non_zero_attributes) == 1:
                relevant_attribute = non_zero_attributes[0]

                w0 = weights[curr_node][0]
                wi = weights[curr_node][relevant_attribute + 1]
                threshold = - w0 / wi

                output += f"{attribute_names[relevant_attribute + 1]} <= {'{:.3f}'.format(threshold)}"

                try:
                    if wi < 0:
                        subtree_depth = depth - curr_depth
                        if subtree_depth > 1:
                            subtree_size = 2**(subtree_depth - 2)
                            swap = weights[curr_node + 1 : curr_node + subtree_size + 1].copy()
                            weights[curr_node + 1 : curr_node + subtree_size + 1] = weights[curr_node + subtree_size + 1 : curr_node + 2 * subtree_size + 1]
                            weights[curr_node + subtree_size + 1 : curr_node + 2 * subtree_size + 1] = swap
                        
                        start_label = curr_node - curr_node % 2
                        swap = labels[start_label : start_label + 2 ** (subtree_depth - 1)].copy()
                        labels[start_label : start_label + 2 ** (subtree_depth - 1)] = labels[start_label + 2 ** (subtree_depth - 1) : start_label + 2 ** (subtree_depth)]                    
                        labels[start_label + 2 ** (subtree_depth - 1) : start_label + 2 ** (subtree_depth)] = swap
                except:
                    pdb.set_trace()
            elif len(non_zero_attributes) == 0:
                output += f"0 <= {'{:.3f}'.format(weights[curr_node][0])}"
            else:
                output += '{:.3f}'.format(weights[curr_node][0]) + " + " + \
                    " + ".join([f"{'{:.3f}'.format(w)} {attribute_names[i]}" for i, w in enumerate(weights[curr_node][1:])])
            
            curr_node += 1
            
            stack.append((node + 1, curr_depth + 1))
            stack.append((node + 2 ** (depth - curr_depth), curr_depth + 1))
        output += "\n"

    return output

def get_penalty(weights, max_penalty=1, alpha=1, should_normalize_rows=False, 
    should_normalize_penalty=False, should_apply_exp=False, kappa=2):

    # b = np.abs(W)
    # b[np.arange(len(b)), np.argmax(b[:, 1:], axis=1) + 1] = 0
    # penalty = np.sum(np.abs(W) - b)

    if should_normalize_rows:
        penalty = np.sum([((np.sum(row[1:] / np.max(row[1:])) - 1) if np.max(row[1:]) > 0 else 0) for row in np.abs(weights)])
    else: 
        penalty = np.sum([np.sum(row[1:]) - np.max(row[1:]) for row in np.abs(weights)])
    
    if should_apply_exp:
        penalty = max_penalty * (1 - np.exp(- kappa * penalty))
    
    if should_normalize_penalty:
        penalty /= max_penalty
    
    return alpha * penalty
    
def calc_accuracy(X, y, W, labels):
    y_pred = predict_batch(X, W, labels, add_1=True)
    acc = np.mean([(1 if y_pred_i == y_i else 0) for y_pred_i, y_i in zip(y, y_pred)])
    
    return acc

def get_W_as_univariate(multiv_W):
    b = np.copy(multiv_W)
    b2 = np.abs(multiv_W)
    b[np.arange(len(b)), np.argmax(b2[:,1:], axis=1) + 1] = 0
    b[:, 0] = 0

    return multiv_W - b

if __name__ == "__main__":
    # W = np.array([[0.228, 0.000, 0.478], [-0.633, 0.384, 0.000], [-0.986, 0.000, 0.043], [0.495, 0.065, 0.000], [-0.691, 0.000, 0.335], [0.065, 0.000, 0.000], [0.927, 0.000, -0.111]])
    # W = np.array([[6.820, -4.568, -5.836], [0.780, 5.135, -4.205], [13.503,  2.909,  1.977]])
    # W = np.array([[2, -1, 0], [-4, 0, 1], [5, -1, 0]])
    # W = np.array([[0.228, 0.000, -0.478], [-0.633, 0.384, 0.000], [-0.986, 0.000, 0.043], [0.495, 0.065, 0.000], [-0.691, 0.000, 0.335], [0.065, 0.000, 0.000], [0.927, 0.000, -0.111]])
    
    # x1 = np.array([-10, 10])
    # x2 = np.array([ 10,  5])
    # x3 = np.array([-10,  0])
    # x4 = np.array([ 0,   0])
    # X = np.stack([x4, x3, x1, x3])

    # y = np.array([0, 1, 2, 1])
    # accuracy, labels = dt_matrix_fit(X, y, W)

    # labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # print(weights2treestr(W, labels))

    X = np.array([[-3, -2], [4, -2], [3, 3], [-3, 1], [-2, -2], [-2, 0]], dtype=np.float64)
    y = np.array([1, 1, 2, 1, 2, 1], dtype=np.float64) - 1
    W = np.array([[2, 3, -4], [5, -5, -4], [6, -2, 3]], dtype=np.float64)
    M = np.array([[-1, -1, 0], [-1, 1, 0], [1, 0, -1], [1, 0, 1]], dtype=np.float64)

    # X = np.array([[-3, -2, 2], [4, -2, 5], [3, 3, 4], [-3, 1, 7], [-2, -2, 9], [-2, 0, 1]])
    # y = np.array([1, 1, 2, 1, 2, 1]) - 1
    # W = np.array([[0.4, 0.2, -0.7, 0.8], [0.2, 0.5, -0.5, 0.4], [-0.6, 0.1, 0.8, -0.5]])
    # M = np.array([[-1, -1, 0], [-1, 1, 0], [1, 0, -1], [1, 0, 1]])

    depth = int(np.log2(len(W) + 1))
    n_leaves = 2**depth
    n_classes = 2
    N = len(X)

    M = create_mask_dx(depth)
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.tile(y, (n_leaves, 1))

    # Compile numba
    dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)

    num_evaluations = 10000
    start_time = time.time()
    for _ in range(num_evaluations):
        dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()
    print(f"(Numba) Elapsed time: {end_time - start_time}")

    start_time = time.time()
    for _ in range(num_evaluations):
        dt_matrix_fit(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()
    print(f"(Old) Elapsed time: {end_time - start_time}")

    start_time = time.time()
    for _ in range(num_evaluations):
        dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()
    print(f"(Dx) Elapsed time: {end_time - start_time}")

    start_time = time.time()
    for _ in range(num_evaluations):
        dt_matrix_fit_dx2(X, y, W, depth, n_classes, X_, Y_, M)
    end_time = time.time()
    print(f"(Dx2) Elapsed time: {end_time - start_time}")

    # Testing whether the evaluation schemes are equal
    for _ in range(1000):
        # n_classes = np.random.randint(2, 5)
        # n_samples = np.random.randint(1000, 10000)
        n_samples = 5
        n_classes = 2
        W = np.random.uniform(-1, 1, (n_leaves - 1, 4))
        X = np.random.uniform(-5, 5, (n_samples, 3))
        y = np.random.randint(0, n_classes, (1, n_samples))
        W = np.round(W, 1)
        X = np.int_(X)

        acc_old, _ = dt_matrix_fit(X, y, W, depth, n_classes)
        acc_dx, _ = dt_matrix_fit_dx(X, y, W, depth, n_classes)
        acc_dx2, _ = dt_matrix_fit_dx2(X, y, W, depth, n_classes)
        acc_numba, _ = dt_matrix_fit_dx_numba(X, y, W, depth, n_classes)
        print(f"ACCURACIES: (old: {acc_old}, dx: {acc_dx}, dx2: {acc_dx2}, numba: {acc_numba})")

        X = np.vstack((np.ones(len(X)), X.T)).T
        Z = np.sign(W @ X.T)
        if np.min(np.max(np.clip(M @ Z - 1, 0, 1), axis=1)) != 0 and acc_old != 1:
            pdb.set_trace()

        if [acc_old, acc_dx, acc_dx2, acc_numba].count(acc_old) != 4:
            pdb.set_trace()
    
    print("Evaluation schemes are the same.")