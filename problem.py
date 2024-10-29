import numpy.typing as npt
import numpy as np
from slicing_tree import (
    Slit as SlicingTree,
    generate_random_slitting_tree,
    swap_random_subtree,
    average_rectangle_size,
    average_variance,
)
from pymoo.problems.multi import ElementwiseProblem

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
        sensors_sheet: npt.NDArray,
    ):
        self.max_rectangle_size = max_rectangle_size
        self.min_rectangle_size = min_rectangle_size
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.sensors_sheet = sensors_sheet

        # TODO: Change n_obj
        # n_var: number of variales in single genotype is 1 slicing tree
        super().__init__(n_var=1, n_obj=3, n_constr=0, type=SlicingTree)

    def _evaluate(self, slitting: list[SlicingTree], out, *args, **kwargs):
        f1 = slitting[0].size
        f2 = average_rectangle_size(
            slitting[0], self.sheet_width, self.sheet_height, self.sensors_sheet
        )
        f3 = average_variance(
            slitting[0], self.sheet_width, self.sheet_height, self.sensors_sheet
        )
        print(f3)
        # f4 = slitting[0].complexity()

        # out["F"] = np.column_stack([f1, -f2, -f3, f4])
        out["F"] = np.column_stack([f1, -f2, -f3])
        # out["F"] = np.column_stack([f1, -f2])

