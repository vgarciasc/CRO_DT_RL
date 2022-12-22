import pdb
import numpy as np
from functools import reduce

from sup_configs import load_dataset, config_CE

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

def dt_matrix_fit(X, y, W, mask=None, X_prepared=None, y_prepared=None, default_label=0):
    num_leaves = len(W) + 1
    depth = int(np.log2(len(W) + 1))

    if mask is not None:
        m = mask
    else:
        # Mask to detect the given leaf for each observation
        m = create_mask(depth=depth)

    if X_prepared is not None:
        X = X_prepared
    else:
        X = np.vstack((np.ones(len(X)).T, X.T)).T

    if y_prepared is not None:
        Y = y_prepared
    else:
        Y = np.tile(y + MAGIC_NUMBER, (num_leaves, 1))

    # Matrix composed of one-hot vectors defining the leaf index of each input
    try:
        K = m @ np.sign(- W @ X.T)
        L = np.clip(K - (np.max(K) - 1), 0, 1)
    except:
        pdb.set_trace()

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

clipper = lambda A, d : reduce(np.multiply, [(i - A) / (i - d) for i in range(-d, d)])
bincount = lambda A, i, k, N : reduce(np.multiply, [(j - A) / (j - i) for j in range(0, k) if j != i]) @ np.ones(N)

def dt_matrix_fit_dx(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    n_leaves = len(W) + 1
    n_attrs = len(W[0]) - 1
    N = len(X)

    M = create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_
    S = np.ones(N)
    
    Z = np.sign(W @ X_.T)
    Z_ = clipper(M @ Z, depth)
    R = Z_ * Y_

    H = np.stack([bincount(R, i, n_classes, N) for i in range(n_classes)])
    H[0] -= (np.ones(n_leaves) * N - Z_ @ S) # removing false zeros

    labels = np.argmax(H, axis=0)
    accuracy = sum(np.max(H, axis=0)) / N
    pdb.set_trace()

    return accuracy, labels

def dt_matrix_fit_dx2(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    n_leaves = len(W) + 1
    n_attrs = len(W[0]) - 1
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
    W = np.array([[0.228, 0.000, -0.478], [-0.633, 0.384, 0.000], [-0.986, 0.000, 0.043], [0.495, 0.065, 0.000], [-0.691, 0.000, 0.335], [0.065, 0.000, 0.000], [0.927, 0.000, -0.111]])
    
    x1 = np.array([-10, 10])
    x2 = np.array([ 10,  5])
    x3 = np.array([-10,  0])
    x4 = np.array([ 0,   0])
    X = np.stack([x4, x3, x1, x3])

    y = np.array([0, 1, 2, 1])
    accuracy, labels = dt_matrix_fit(X, y, W)

    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    print(weights2treestr(W, labels))
