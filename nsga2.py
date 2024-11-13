from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from slicing_tree import (
    Slit as SlicingTree,
    plot_slicing_tree,
    EndNode,
    generate_random_slitting_tree,
    swap_random_subtree,
    average_rectangle_size,
    average_weighted_worst_percentile,
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
from anytree import LevelOrderIter
from data import load_data
from pymoo.visualization.scatter import Scatter
from pymoo.core.duplicate import ElementwiseDuplicateElimination


POPULATION_SIZE = 500
GENERATIONS = 100


class CoilSlitting(ElementwiseProblem):
    def __init__(
        self,
        *,
        sensors_sheet: npt.NDArray,
    ):
        self.sensors_sheet = sensors_sheet

        sheet_width, sheet_height = sensors_sheet.shape

        self.min_rectanle_width = int(0.2 * sheet_width)
        self.min_rectanle_height = int(0.2 * sheet_height)

        self.max_number_of_slits = 5 * 5  # 20% width and height

        super().__init__(n_var=1, n_obj=2, n_constr=0, type=SlicingTree)

    # Objectives:
    # 1. Minimise the average weighted 95 percentile
    # 2. Maximise the average size of the rectangles
    def _evaluate(self, slitting: list[SlicingTree], out, *args, **kwargs):
        # f1 = lambda r: 1 if r is not None else 0
        f1 = average_weighted_worst_percentile(
            slitting[0], sensors_sheet=self.sensors_sheet
        )
        f2 = average_rectangle_size(slitting[0], self.sensors_sheet)

        # f2 = -average_rectangle_size(slitting[0], self.sensors_sheet)
        # print(f"Tree size:{(slitting[0]).size}: {f1=}, {f2=}")

        out["F"] = np.column_stack([f1, -f2])


class RandomSlicingTreeSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.array(
            [
                (
                    generate_random_slitting_tree(
                        size=random.randint(2, problem.max_number_of_slits),
                    ),
                )
                for _ in range(n_samples)
            ]
        )

        return X


class SwapSubtreeCrossover(Crossover):
    def __init__(self, prob: float):
        super().__init__(2, 2, prob=prob)

    def _do(self, problem, X, **kwargs):
        return X

        print("hu")
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            parent1, parent2 = X[0, k, 0], X[1, k, 0]

            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)

            swap_random_subtree(child1, child2)

            Y[0, k, 0], Y[1, k, 0] = child1, child2

        return Y


class JitterMutation(Mutation):
    def __init__(self, prob: float):
        super().__init__(prob=prob)

    def _do(self, problem, X, **kwargs):
        return X

        for i in range(len(X)):
            tree = X[i, 0]
            for node in tree.descendants:
                if isinstance(node, EndNode):
                    continue

                if random.uniform(0, 1) < 1 / 3:
                    node.offset = max(0, min(1, node.offset + random.gauss(0, 0.15)))

        return X


class DuplicateElimination(ElementwiseDuplicateElimination):
    # TODO: compute hash beforehand and compare
    def is_equal(self, a, b):
        tree1 = a.X[0]
        tree2 = b.X[0]

        if tree1.size != tree2.size:
            return False

        for node_a, node_b in zip(LevelOrderIter(tree1), LevelOrderIter(tree2)):
            if isinstance(node_a, EndNode) and isinstance(node_b, EndNode):
                continue

            if (isinstance(node_a, EndNode) and not isinstance(node_b, EndNode)) or (
                isinstance(node_b, EndNode) and not isinstance(node_a, EndNode)
            ):
                return False

        #     if node_a.offset != node_b.offset or node_a.horizontal != node_b.horizontal:
        #         return False

        print("Equal")
        return True


if __name__ == "__main__":

    single_sheet = load_data("./data.csv")[0]

    sheet_width = 100
    sheet_height = 100

    min_rectangle_width = 0.2 * sheet_width
    min_rectangle_height = 0.2 * sheet_height

    problem = CoilSlitting(
        sensors_sheet=single_sheet,
    )
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        # eliminate_duplicates=DuplicateElimination(),
        eliminate_duplicates=NoDuplicateElimination(),
        sampling=RandomSlicingTreeSampling(),
        crossover=SwapSubtreeCrossover(prob=0.5),
        mutation=JitterMutation(prob=0.8),
    )
    res = minimize(
        problem, algorithm, ("n_gen", GENERATIONS), seed=0xC0FFEE, verbose=True
    )
    print(res.X.shape)
    # print(x1, x2)
    print(res.X)
    # plot_slicing_tree(x1)
    # plot_slicing_tree(x2)
    print()
    # print(problem.pareto_front())

    print(res.F)
    print(np.unique(res.F, axis=0).size)
    Scatter().add(res.F).show()

#     for i in range(len(res.X)):
#         print(res.X[i])
#         print(res.F[i])
#         print()
#         plot_slicing_tree(res.X[i][0])
