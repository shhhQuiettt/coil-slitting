from dataclasses import dataclass
import numpy.typing as npt
from slicing_tree import (
    generate_random_slitting_tree,
    swap_random_subtree,
)
import numpy as np
from copy import deepcopy
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.core.duplicate import NoDuplicateElimination
from anytree import NodeMixin
from data import load_data
from problem import CoilSlitting

POPULATION_SIZE = 1000
GENERATIONS = 10
MAX_NUMBER_OF_SLITS = 20

class RandomSlicingTreeSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # X = np.full((n_samples, 1), None, dtype=object)

        # for individual_id in range(n_samples):
        #     # doesnt include minimal rectangle size
        #     X[individual_id, 0] = SlicingTree.random(MAX_NUMBER_OF_SLITS)
        X = np.array(
            [
                (generate_random_slitting_tree(random.randint(2, MAX_NUMBER_OF_SLITS)),)
                for _ in range(n_samples)
            ]
        )

        return X


class SwapSubtreeCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            parent1, parent2 = X[0, k, 0], X[1, k, 0]

            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)

            swap_random_subtree(child1, child2)

            Y[0, k, 0], Y[1, k, 0] = child1, child2

        return Y


class AddSlittingAndNudgeMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        return X


if __name__ == "__main__":

    single_sheet = load_data("./data.csv")[0]

    problem = CoilSlitting(
        max_rectangle_size=-1,
        min_rectangle_size=-1,
        sheet_width=100,
        sheet_height=100,
        sensors_sheet=single_sheet,
    )
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        eliminate_duplicates=NoDuplicateElimination(),
        sampling=RandomSlicingTreeSampling(),
        crossover=SwapSubtreeCrossover(),
        mutation=AddSlittingAndNudgeMutation(),
    )
    res = minimize(
        problem, algorithm, ("n_gen", GENERATIONS), seed=0xC0FFEE, verbose=True
    )
    print(res.X)
