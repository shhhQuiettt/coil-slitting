from dataclasses import dataclass
import numpy as np
from pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort
from data import load_data
import matplotlib.pyplot as plt
import numpy.typing as npt
import random
from typing import Any
from anytree import NodeMixin, PreOrderIter
from anytree.exporter import DotExporter
import tempfile
import cv2
from queue import Queue


@dataclass
class Slit(NodeMixin):
    horizontal: bool
    offset: float  # percentage of the width/height

    def __repr__(self) -> str:
        return f"Slit(horizontal={self.horizontal}, offset={self.offset:.2f}, size={self.size})"

    def __str__(self) -> str:
        return f"{'H' if self.horizontal else 'V'}  {self.offset:.3f}"


# class to represent the end node, to know if a subrectangle is left or right
class EndNode(NodeMixin):
    # for some reason to display graph with graphviz
    # every node shoul have unique identifier that ALSO is what is displayed
    count = 0

    def __init__(self):
        self.count = EndNode.count + 1
        EndNode.count += 1
        super().__init__()

    def __str__(self) -> str:
        return f"END{self.count}"


@dataclass
class Rectangle:
    width: float
    height: float
    sensors: npt.NDArray

    def sensors_variance(self) -> float:
        return self.sensors.var()


def generate_random_slitting_tree(size: int) -> Slit:
    if size == 1:
        return Slit(horizontal=random.choice([True, False]), offset=random.random())

    size -= 1

    left_size = random.randint(0, size)
    right_size = size - left_size

    children = []
    if left_size > 0:
        children_left = (generate_random_slitting_tree(left_size),)
        children += children_left

    else:
        children.append(EndNode())

    if right_size > 0:
        children_right = (generate_random_slitting_tree(right_size),)
        children += children_right
    else:
        children.append(EndNode())

    node = Slit(
        horizontal=random.choice([True, False]),
        offset=random.random(),
    )

    node.children = children
    return node


def swap_random_subtree(tree1: Slit, tree2: Slit) -> None:
    node1 = random.choice(tree1.descendants)
    node2 = random.choice(tree2.descendants)

    node1.parent, node2.parent = node2.parent, node1.parent


def get_rectangles(
    node: Slit,
    width: float,
    height: float,
    sensors_sheet: npt.NDArray,
) -> list[Rectangle]:
    if isinstance(node, EndNode):
        return []

    height1 = height * node.offset if node.horizontal else height
    height2 = height - height1 if node.horizontal else height

    width1 = width * node.offset if not node.horizontal else width
    width2 = width - width1 if not node.horizontal else width

    if not node.children:

        return [
            # what about sensors on the edge?
            Rectangle(width1, height1, sensors_sheet[: int(width1), : int(height1)]),
            Rectangle(width2, height2, sensors_sheet[int(width1) :, int(height1) :]),
        ]

    rectangles_children1 = get_rectangles(
        node.children[0], width1, height1, sensors_sheet[: int(width1), : int(height1)]
    )

    if len(node.children) == 1:
        return rectangles_children1

    rectangles_children2 = get_rectangles(
        node.children[1], width2, height2, sensors_sheet[int(width1) :, int(height1) :]
    )

    return rectangles_children1 + rectangles_children2


def average_rectangle_size(
    tree: Slit, sheet_width: int, sheet_height: int, sensors_sheet: npt.NDArray
) -> float:
    rectangles = get_rectangles(tree, sheet_width, sheet_height, sensors_sheet)
    return sum(rectangle.width * rectangle.height for rectangle in rectangles) / len(
        rectangles
    )


def average_variance(
    tree: Slit, sheet_width: int, sheet_height: int, sensors_sheet: npt.NDArray
):
    rectangles = get_rectangles(tree, sheet_width, sheet_height, sensors_sheet)

    variances = [rectangle.sensors_variance() for rectangle in rectangles]
    return sum(rectangle.sensors_variance() for rectangle in rectangles) / len(
        rectangles
    )


def plot_slicing_tree(tree: Slit):
    with tempfile.NamedTemporaryFile(suffix=".png") as dot_output:
        DotExporter(
            tree,
            nodenamefunc=lambda node: str(node),
        ).to_picture(dot_output.name)
        plt.imshow(plt.imread(dot_output.name))
        plt.show()


def plot_slits(tree: Slit, sensors_sheet: npt.NDArray):
    # rectangles = get_rectangles(tree, 100, 100, sensors_sheet)

    image = sensors_sheet.copy()
    # image = np.pad(image, 5)

    range_h = [0, sensors_sheet.shape[0]]
    range_v = [0, sensors_sheet.shape[1]]

    range_h_stack = Queue()
    range_v_stack = Queue()

    for slit in PreOrderIter(tree):
        if isinstance(slit, EndNode):
            continue

        if slit.horizontal:
            range_h_stack.put(range_h)
            range_h = [range_h[0], int(slit.offset * range_h[1])]
        else:
            range_v_stack.put(range_v)
            range_v = [range_v[0], int(slit.offset * range_v[1])]

    for slit in PreOrderIter(tree):
        if isinstance(slit, EndNode):
            continue

        if slit.horizontal:
            line_x = int(slit.offset * range_h[1])
            image[line_x, :] = 0

        else:
            line_y = int(slit.offset * range_v[1])
            # image[:, line_y] =

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    example_tree = generate_random_slitting_tree(3)

    np.random.seed(1)
    print(np.random.random((100, 100)))
    for r in get_rectangles(example_tree, 100, 100, np.random.random((100, 100))):
        print(r.width, r.height)
        print(r.sensors)
        print()
    # sheet = load_data("./data.csv")[0]
    # plot_slits(example_tree, sheet)
    # plot_slicing_tree(example_tree)
