from dataclasses import dataclass
from slicing_tree import (
    Slit as SlicingTree,
    generate_random_slitting_tree,
    swap_random_subtree,
    average_rectangle_size,
)
import numpy as np
from copy import deepcopy
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.problems.multi import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.duplicate import NoDuplicateElimination
from anytree import NodeMixin

POPULATION_SIZE = 1000
MAX_NUMBER_OF_SLITS = 20


# Objectives:
# 1. Minimize the number of slits
# 2. Maximize the average size of the slits
# 3. Maximize the average quality of the slits
# 4. Minimize the complexity of the slits
class CoilSlitting(ElementwiseProblem):
    def __init__(
        self,
        *,
        sheet_width: int,
        sheet_height: int,
        max_rectangle_size: float,
        min_rectangle_size: float,
    ):
        self.max_rectangle_size = max_rectangle_size
        self.min_rectangle_size = min_rectangle_size
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        # TODO: Change n_obj
        super().__init__(n_var=1, n_obj=2, n_constr=0, type=SlicingTree)

    def _evaluate(self, slitting: list[SlicingTree], out, *args, **kwargs):
        f1 = slitting[0].size
        f2 = average_rectangle_size(slitting[0], self.sheet_width, self.sheet_height)
        # f3 = slitting[0].average_slit_quality()
        # f4 = slitting[0].complexity()

        # out["F"] = np.column_stack([f1, -f2, -f3, f4])
        out["F"] = np.column_stack([f1, -f2])


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
    problem = CoilSlitting(
        max_rectangle_size=-1, min_rectangle_size=-1, sheet_width=100, sheet_height=100
    )
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        eliminate_duplicates=NoDuplicateElimination(),
        sampling=RandomSlicingTreeSampling(),
        crossover=SwapSubtreeCrossover(),
        mutation=AddSlittingAndNudgeMutation(),
    )
    res = minimize(problem, algorithm, ("n_gen", 100), seed=0xC0FFEE, verbose=True)
    print(res.X)
