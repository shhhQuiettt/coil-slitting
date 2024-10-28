from dataclasses import dataclass
import numpy as np
from copy import deepcopy
import random
from typing import Optional
from numpy._typing import ArrayLike
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.problems.multi import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.duplicate import NoDuplicateElimination

POPULATION_SIZE = 10
MAX_NUMBER_OF_SLITS = 10


@dataclass
class Slit:
    horizontal: bool
    offset: float  # percentage of the width/height
    left_slit: Optional["Slit"]
    right_slit: Optional["Slit"]

    def is_leaf(self) -> bool:
        return self.left_slit is None and self.right_slit is None


class SlicingTree:
    def __init__(self, number_of_slits: int = 0, root: Optional[Slit] = None):
        self.number_of_slits = number_of_slits
        self.root: Optional[Slit] = root

    def average_slit_size(self) -> float:
        return 0

    def average_slit_quality(self) -> float:
        return 0

    def complexity(self) -> float:
        return 0

    def __repr__(self):
        return f"SlicingTree(number_of_slits={self.number_of_slits}, root={self.root is not None})"

    @classmethod
    def random(cls, max_number_of_slits: int) -> "SlicingTree":
        no_of_slits = np.random.randint(1, max_number_of_slits)
        root = generate_random_slitting_tree_root(no_of_slits)

        return cls(no_of_slits, root)

    def get_random_node(self) -> Optional[Slit]:
        current = self.root
        assert current is not None

        while True:
            if random.choice([True, False]):
                current = current.left_slit
            else:
                current = current.right_slit

            if random.random() < 0.2 or current is None:
                return current

    def get_random_leaf(self) -> Optional[Slit]:
        current = self.root
        while current is not None:
            if current.left_slit is None and current.right_slit is None:
                return current
            if current.left_slit is None:
                current = current.right_slit
                continue

            elif current.right_slit is None:
                current = current.left_slit
                continue

            if random.choice([True, False]):
                current = current.left_slit
            else:
                current = current.right_slit

    def add_random_slit(self):
        leaf = self.get_random_leaf()
        if leaf is None:
            return
        if random.choice([True, False]):
            leaf.left_slit = Slit(
                horizontal=random.choice([True, False]),
                offset=random.random(),
                left_slit=None,
                right_slit=None,
            )
        else:
            leaf.right_slit = Slit(
                horizontal=random.choice([True, False]),
                offset=random.random(),
                left_slit=None,
                right_slit=None,
            )

    def remove_random_leaf(self):
        current = self.root
        assert current is not None
        assert not current.is_leaf()

        cut_right = random.choice([True, False])

        if cut_right and current.right_slit is None:
            return
        if not cut_right and current.left_slit is None:
            return

        while True:
            if cut_right:
                if current.right_slit is None:
                    return

                if current.right_slit.is_leaf():
                    current.right_slit = None
                    return
            else:
                if current.left_slit is None:
                    return

                if current.left_slit.is_leaf():
                    current.left_slit = None
                    return

            if random.random() < 0.5:
                current = current.left_slit
            else:
                current = current.right_slit


# Objectives:
# 1. Minimize the number of slits
# 2. Maximize the average size of the slits
# 3. Maximize the average quality of the slits
# 4. Minimize the complexity of the slits
class CoilSlitting(ElementwiseProblem):
    def __init__(self, max_rectangle_size: float, min_rectangle_size: float):
        self.max_rectangle_size = max_rectangle_size
        self.min_rectangle_size = min_rectangle_size
        super().__init__(n_var=1, n_obj=4, n_constr=0, type=SlicingTree)

    def _evaluate(self, slitting: list[SlicingTree], out, *args, **kwargs):
        f1 = slitting[0].number_of_slits
        f2 = slitting[0].average_slit_size()
        f3 = slitting[0].average_slit_quality()
        f4 = slitting[0].complexity()

        out["F"] = np.column_stack([f1, -f2, -f3, f4])


def generate_random_slitting_tree_root(size: int) -> Optional[Slit]:
    if size == 0:
        return None

    size -= 1

    left_size = random.randint(0, size)
    right_size = size - left_size

    return Slit(
        horizontal=random.choice([True, False]),
        offset=random.random(),
        left_slit=generate_random_slitting_tree_root(left_size),
        right_slit=generate_random_slitting_tree_root(right_size),
    )


class RandomSlicingTreeSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        for individual_id in range(n_samples):
            # doesnt include minimal rectangle size
            X[individual_id, 0] = SlicingTree.random(MAX_NUMBER_OF_SLITS)

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

            node1 = child1.get_random_node()
            node2 = child2.get_random_node()

            if node1 is not None and node2 is not None:
                node1.left_slit, node2.left_slit = node2.left_slit, node1.left_slit
                node1.right_slit, node2.right_slit = node2.right_slit, node1.right_slit

            Y[0, k, 0], Y[1, k, 0] = child1, child2

        return Y


class AddSlittingAndNudgeMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):

            node = X[i, 0].get_random_node()
            if node is not None:
                node.offset += np.random.normal(0, 0.2)
                node.offset = max(0, min(1, node.offset))

            if random.random() < 0.5:
                X[i, 0].add_random_slit()
            else:
                X[i, 0].remove_random_leaf()

        return X


if __name__ == "__main__":
    problem = CoilSlitting(max_rectangle_size=-1, min_rectangle_size=-1)
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        eliminate_duplicates=NoDuplicateElimination(),
        sampling=RandomSlicingTreeSampling(),
        crossover=SwapSubtreeCrossover(),
        mutation=AddSlittingAndNudgeMutation(),
    )
    res = minimize(problem, algorithm, ("n_gen", 2), seed=0xC0FFEE)
    print(res.F)
