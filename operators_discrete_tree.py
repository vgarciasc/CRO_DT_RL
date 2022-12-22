import pdb
import numpy as np

def switch_attribute(solution, n_attributes, n_splits_to_modify):
    splits_idx = list(range(len(solution)))
    np.random.shuffle(splits_idx)

    for split_idx in splits_idx[:n_splits_to_modify]:
        new_attrib = np.random.randint(0, n_attributes)
        solution[split_idx][1:] = np.zeros(n_attributes)
        solution[split_idx][new_attrib + 1] = 1

    return solution

def modify_threshold(solution, strength=0.5):
    n_splits = len(solution)
    random_split_idx = np.random.randint(n_splits)
    solution[random_split_idx][0] += np.random.normal(0, strength)
    return solution

def reset_threshold(solution):
    n_splits = len(solution)
    random_split_idx = np.random.randint(n_splits)
    solution[random_split_idx][0] = np.random.uniform(-1, 1)
    return solution

def reset_split(solution, n_attributes):
    n_splits = len(solution)

    random_split_idx = np.random.randint(n_splits)
    new_attrib = np.random.randint(0, n_attributes)

    solution[random_split_idx][0] = np.random.uniform(-1, 1)
    solution[random_split_idx][1:] = np.zeros(n_attributes)
    solution[random_split_idx][new_attrib + 1] = 1

    return solution

if __name__ == "__main__":
    solution = np.array([[0.5, 0, 1, 0], [0.2, 1, 0, 0], [-0.3, 0, 0, 1]])
    # solution = switch_attribute(solution, 2)
    modify_threshold(solution)
    print(solution)