import pdb
import numpy as np
from copy import deepcopy

MAGIC_NUMBER = 42

class SoftTree:
    def __init__(self, num_attributes=2, num_classes=2, depth=2, weights=[], labels=[]):
        self.num_attributes = num_attributes
        self.num_classes = num_classes
        self.weights = weights
        self.labels = labels
        
        self.num_nodes = len(weights)
        self.num_leaves = len(labels)
        self.depth = depth

        self.mask = self.create_mask()
    
    def randomize(self, depth):
        self.num_nodes = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        self.weights = np.random.uniform(-1, 1, size=(self.num_nodes, self.num_attributes + 1))
        self.labels = np.ones((self.num_leaves, self.num_classes)) / self.num_classes

    def get_left(self, node):
        return node * 2 + 1
    
    def get_right(self, node):
        return node * 2 + 2
    
    def is_leaf(self, node):
        return node >= self.num_nodes

    def get_leaf(self, state):
        state = np.insert(state, 0, 1, axis=0)

        stack = [0]

        while stack != []:
            node = stack.pop(0)

            if self.is_leaf(node):
                return node - self.num_nodes
            else:
                if self.weights[node] @ state <= 0:
                    stack.append(self.get_left(node))
                else:
                    stack.append(self.get_right(node))
    
    def update_leaves_by_dataset(self, X, y):
        count = [np.zeros(self.num_classes) for _ in range(self.num_leaves)]
        
        for x_i, y_i in zip(X, y):
            leaf = self.get_leaf(x_i)
            count[leaf][y_i] += 1
        
        self.labels = np.zeros((self.num_leaves, self.num_classes))
        for leaf, samples in enumerate(count):
            self.labels[leaf][np.argmax(samples)] = 1

    def predict(self, state):
        return np.argmax(self.labels[self.get_leaf(state)])

    def act(self, state):
        return self.predict(state)

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])
    
    def predict_batch_matrix(self, X):
        X = np.vstack((np.ones(len(X)).T, X.T)).T

        K = self.mask @ np.sign(- self.weights @ X.T)
        L = np.clip(K - (np.max(K) - 1), 0, 1)
        leaves = np.argmax(L.T, axis=1)
        y = np.array([self.labels[l] for l in leaves])

        return y

    def dt_matrix_fit(self, X, y, default_label=0):
        W = self.weights
        
        num_leaves = len(W) + 1

        X = self.X_ 
        Y = self.Y_

        # Mask to detect the given leaf for each observation
        m = self.mask

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
    
    def create_mask(self):
        depth = self.depth
        
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
        
    def get_used_split_mask(self):
        stack = [(0, 1)]
        output = []

        while len(stack) > 0:
            node, depth = stack.pop(0)

            left = self.get_left(node)
            right = self.get_right(node)

            if not self.is_leaf(node) and self.is_leaf(left) and self.is_leaf(right):
                if self.labels[left - self.num_nodes] == self.labels[right - self.num_nodes]:
                    output.append(1)
                else:
                    output.append(0)
            else:
                bias = self.weights[node][0]
                attribute = np.argmax(np.abs(self.weights[node][1:])) + 1
                weight = self.weights[node][attribute]

                output += f"x{attribute} <= {'{:.3f}'.format(-bias / weight)}"
                
                if (weight < 0):
                    stack.append((self.get_left(node), depth + 1))
                    stack.append((self.get_right(node), depth + 1))
                else:
                    stack.append((self.get_right(node), depth + 1))
                    stack.append((self.get_left(node), depth + 1))
            output += "\n"

        return output
    
    def turn_univariate(self):
        self.weights = np.array([[(w if i == 0 or i == (np.argmax(np.abs(split[1:])) + 1) else 0) for i, w in enumerate(split)] for split in self.weights])
        
        new_labels = np.zeros(self.labels.shape)
        for leaf in range(self.num_leaves):
            new_labels[leaf][np.argmax(self.labels[leaf])] = 1
        self.labels = new_labels
    
    def str_univariate(self, config=None):
        stack = [(0, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if self.is_leaf(node):
                action = np.argmax(self.labels[node - self.num_nodes])
                output += f"Class {action}" if config is None else config["actions"][action]
            else:
                bias = self.weights[node][0]
                attribute = np.argmax(np.abs(self.weights[node][1:])) + 1
                weight = self.weights[node][attribute]

                if config is None:
                    output += f"x{attribute} <= {'{:.3f}'.format(-bias / weight)}"
                else:
                    output += f"{config['attributes'][attribute- 1][0]} <= {'{:.3f}'.format(-bias / weight)}"
                
                if (weight < 0):
                    stack.append((self.get_left(node), depth + 1))
                    stack.append((self.get_right(node), depth + 1))
                else:
                    stack.append((self.get_right(node), depth + 1))
                    stack.append((self.get_left(node), depth + 1))
            output += "\n"

        return output
    
    def get_splits(self):
        if self.is_leaf():
            return []
        return [self] + self.left.get_splits() + self.right.get_splits()

    def copy(self):
        return deepcopy(self)

    def __str__(self):
        stack = [(0, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if self.is_leaf(node):
                output += str(self.labels[node - self.num_nodes])
            else:
                output += '{:.3f}'.format(self.weights[node][0]) + " + " + \
                    " + ".join([f"{'{:.3f}'.format(self.weights[node][i])} x{i}" for i in range(1, self.num_attributes + 1)])
                
                stack.append((self.get_right(node), depth + 1))
                stack.append((self.get_left(node), depth + 1))
            output += "\n"

        return output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SoftTreeSigmoid(SoftTree):
    def get_leaf(self, state):
        state = np.insert(state, 0, 1, axis=0)

        stack = [(0, 1)]
        output = []

        while stack != []:
            node, membership = stack.pop(0)

            if self.is_leaf(node):
                output.append((node - self.num_nodes, membership))
            else:
                val = sigmoid(self.weights[node] @ state)
                stack.append((self.get_right(node), membership * (1 - val)))
                stack.append((self.get_left(node), membership * val))
        
        max_leaf, max_membership = max(output, key=lambda x:x[1])
        return max_leaf