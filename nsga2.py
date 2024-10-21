from dataclasses import dataclass
import numpy as np
from typing import Optional
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi import ElementwiseProblem


class SlicingTree:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # initiate the root with the full width
        self.number_of_slits = 0
        self.root: Optional[Slit] = None

    def average_slit_size(self) -> float:
        return 0

    def average_slit_quality(self) -> float:
        return 0

    def complexity(self) -> float:
        return 0


@dataclass
class Slit:
    horizontal: bool
    offset: int
    next_slit: Optional["Slit"]


# Objectives:
# 1. Minimize the number of slits
# 2. Maximize the average size of the slits
# 3. Maximize the average quality of the slits
# 4. Minimize the complexity of the slits
class CoilSlitting(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=4, n_constr=0, type=SlicingTree)

    def _evaluate(self, slitting: SlicingTree, out, *args, **kwargs):
        f1 = slitting.number_of_slits
        f2 = slitting.average_slit_size()
        f3 = slitting.average_slit_quality()
        f4 = slitting.complexity()

        out["F"] = np.column_stack([f1, -f2, -f3, f4])
