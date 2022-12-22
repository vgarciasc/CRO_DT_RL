import math
import pdb
import random
import numpy as np
from SoftTree import SoftTree

def DERand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)
        v = r1.solution.weights + F*(r2.solution.weights - r3.solution.weights)
        mask = np.random.random(v.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DEBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = best.solution.weights + F*(r1.solution.weights-r2.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DERand2(solution, population, F, CR):
    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.solution.weights + F*(r2.solution.weights-r3.solution.weights) + F*(r4.solution.weights-r5.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DEBest2(solution, population, F, CR):
    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.solution.weights + F*(r1.solution.weights-r2.solution.weights) + F*(r3.solution.weights-r4.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DECurrentToBest1(solution, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = solution.weights + F*(best.solution.weights-solution.weights) + F*(r1.solution.weights-r2.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DECurrentToRand1(solution, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = solution.weights + np.random.random()*(r1.solution.weights-solution.weights) + F*(r2.solution.weights-r3.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

def DECurrentToPBest1(solution, population, F, CR, p=0.11):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        pbest_idx = random.choice(np.argsort(fitness)[:math.ceil(len(population)*p)])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = solution.weights + F*(pbest.solution.weights-solution.weights) + F*(r1.solution.weights-r2.solution.weights)
        mask = np.random.random(solution.weights.shape) <= CR
        solution.weights[mask] = v[mask]
    return solution

if __name__ == "__main__":
    n_attribs, n_classes = 10, 5
    soft_tree = SoftTree(n_attribs, n_classes).randomize(depth=2)
    population = [
        SoftTree(n_attribs, n_classes).randomize(depth=2),
        SoftTree(n_attribs, n_classes).randomize(depth=2),
        SoftTree(n_attribs, n_classes).randomize(depth=2),
        SoftTree(n_attribs, n_classes).randomize(depth=2)
    ]

    DERand1(soft_tree, population, 1, 1)