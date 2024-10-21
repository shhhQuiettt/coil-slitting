from dataclasses import dataclass
import numpy as np
import random
from typing import Optional
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.problems.multi import ElementwiseProblem

MAX_NUMBER_OF_SLITS = 10


@dataclass
class Slit:
    horizontal: bool
    offset: float  # percentage of the width/height
    left_slit: Optional["Slit"]
    right_slit: Optional["Slit"]


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

    @classmethod
    def random(cls, max_number_of_slits: int) -> "SlicingTree":
        no_of_slits = np.random.randint(0, max_number_of_slits)
        root = generate_random_slitting_tree_root(no_of_slits)
        return cls(no_of_slits, root)


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

    def _evaluate(self, slitting: SlicingTree, out, *args, **kwargs):
        f1 = slitting.number_of_slits
        f2 = slitting.average_slit_size()
        f3 = slitting.average_slit_quality()
        f4 = slitting.complexity()

        out["F"] = np.column_stack([f1, -f2, -f3, f4])


def generate_random_slitting_tree_root(size: int) -> Optional[Slit]:
    if size == 0:
        return None

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

        for i in range(n_samples):
            # doesnt include minimal rectangle size
            X[i, 0] = SlicingTree.random(MAX_NUMBER_OF_SLITS)
