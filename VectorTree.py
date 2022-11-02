import pdb
import numpy as np

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

def get_accuracy(X, y, W, default_label=0):
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

def weights2treestr(weights, labels, data_config=None, use_attribute_names=False):
    if data_config is not None and use_attribute_names:
        attribute_names = data_config["attributes"]
    else:
        attribute_names = [f"x{i}" for i in range(len(weights[0]))]

    depth = int(np.log2(len(weights) + 1)) + 1
    stack = [(0, 1)]
    output = ""
    curr_leaf = 0
    curr_node = 0

    while len(stack) > 0:
        node, curr_depth = stack.pop()
        output += "-" * curr_depth + " "

        if curr_depth == depth:
            output += str(int(labels[curr_leaf]))
            curr_leaf += 1
        else:
            non_zero_attributes = [i for i, w in enumerate(weights[curr_node][1:]) if np.abs(w) > 0.001]
            if len(non_zero_attributes) == 1:
                relevant_attribute = non_zero_attributes[0]
                w0 = weights[curr_node][0]
                wi = weights[curr_node][relevant_attribute]
                
                if wi < 0:
                    swap = labels[curr_leaf]
                    labels[curr_leaf] = labels[curr_leaf + 1]
                    labels[curr_leaf + 1] = swap

                output += f"{attribute_names[relevant_attribute]} <= {'{:.3f}'.format(- w0 / wi)}"
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

if __name__ == "__main__":
    W = np.array([[0.228, 0.000, 0.478], [-0.633, 0.384, 0.000], [-0.986, 0.000, 0.043], [0.495, 0.065, 0.000], [-0.691, 0.000, 0.335], [0.065, 0.000, 0.000], [0.927, 0.000, -0.111]])
    # W = np.array([[6.820, -4.568, -5.836], [0.780, 5.135, -4.205], [13.503,  2.909,  1.977]])
    
    x1 = np.array([-10, 10])
    x2 = np.array([ 10,  5])
    x3 = np.array([-10,  0])
    x4 = np.array([ 0,   0])
    X = np.stack([x4, x3, x1, x3])

    y = np.array([0, 1, 2, 1])
    accuracy, labels = get_accuracy(X, y, W)

    print(weights2treestr(W, labels))
