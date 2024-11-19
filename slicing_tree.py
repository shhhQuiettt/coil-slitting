from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
from data import load_data, display_sheet
import random
from anytree import NodeMixin, PreOrderIter
from anytree.exporter import DotExporter
import tempfile
from queue import Queue


@dataclass
class Slit(NodeMixin):
    horizontal: bool
    offset: float  # percentage of the width/height
    # offset: int  # absolute value

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
    sensors: npt.NDArray

    def sensors_variance(self) -> float:
        return self.sensors.var()

    @property
    def width(self) -> int:
        return self.sensors.shape[0]

    @property
    def height(self) -> int:
        return self.sensors.shape[1]


def generate_random_slitting_tree(
    size: int,
) -> Slit:
    if size == 1:
        return Slit(
            horizontal=random.choice([True, False]), offset=random.uniform(0.3, 0.7)
        )
        # return Slit()

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
        # offset=random.random(),
        offset=random.uniform(0.3, 0.7),
    )

    node.children = children
    return node


def swap_random_subtree(tree1: Slit, tree2: Slit) -> None:
    node1 = random.choice(tree1.descendants)
    node2 = random.choice(tree2.descendants)

    node1.parent, node2.parent = node2.parent, node1.parent


def split_array(
    sensors: npt.NDArray,
    offset: float,
    horizontal: bool,
) -> tuple[npt.NDArray, npt.NDArray]:
    if horizontal:
        return (
            sensors[: int(offset * sensors.shape[0])],
            sensors[int(offset * sensors.shape[0]) :],
        )
    else:
        return (
            sensors[:, : int(offset * sensors.shape[1])],
            sensors[:, int(offset * sensors.shape[1]) :],
        )


# TODO: return np.array
def get_rectangles(
    node: Slit,
    sensors_sheet: npt.NDArray,
) -> list[Rectangle]:
    if isinstance(node, EndNode):
        return []

    sensors1, sensors2 = split_array(sensors_sheet, node.offset, node.horizontal)

    if not node.children:
        return [
            Rectangle(sensors=sensors1),
            Rectangle(sensors=sensors2),
        ]

    assert len(node.children) == 2
    res = []
    if isinstance(node.children[0], EndNode):
        res.append(Rectangle(sensors=sensors1))
    else:
        res.extend(get_rectangles(node.children[0], sensors1))

    if isinstance(node.children[1], EndNode):
        res.append(Rectangle(sensors=sensors2))
    else:
        res.extend(get_rectangles(node.children[1], sensors2))

    return res


def average_rectangle_size(tree: Slit, sensors_sheet: npt.NDArray) -> float:
    rectangles = get_rectangles(tree, sensors_sheet)
    # assert np.all([rectangle.sensors.size > 0 for rectangle in rectangles])
    return sum(rectangle.width * rectangle.height for rectangle in rectangles) / len(
        rectangles
    )


def average_variance(tree: Slit, sensors_sheet: npt.NDArray):
    rectangles = get_rectangles(tree, sensors_sheet)

    return sum(rectangle.sensors_variance() for rectangle in rectangles) / len(
        rectangles
    )


def average_weighted_worst_percentile(
    tree: Slit, *, sensors_sheet: npt.NDArray, percentile: float = 0.95
) -> float:
    rectangles = get_rectangles(tree, sensors_sheet)
    # assert non-empty
    # assert np.all([rectangle.sensors.size > 0 for rectangle in rectangles])

    percentiles = np.array(
        [
            np.percentile(r.sensors, percentile * 100) if r.sensors.size > 0 else 0
            for r in rectangles
        ]
    )
    areas = np.array([rectangle.width * rectangle.height for rectangle in rectangles])

    return np.sum(percentiles * areas) / np.sum(areas)


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


def make_rectangles_in_sheet(node: Slit, sheet: npt.NDArray) -> npt.NDArray:
    if isinstance(node, EndNode):
        return sheet

    sensors1, sensors2 = split_array(sheet, node.offset, node.horizontal)

    if node.is_leaf:
        s1 = sensors1
        s2 = sensors2
    else:
        s1 = make_rectangles_in_sheet(node.children[0], sensors1)
        s2 = make_rectangles_in_sheet(node.children[1], sensors2)

    if node.horizontal:
        # zero_row = np.zeros((1, sensors1.shape[1]))
        # sensors1 = np.vstack([sensors1, zero_row])
        # change last row to zero
        s1[-1, :] = 0
        return np.vstack([s1, s2])
    else:
        # zero_column = np.zeros((sensors1.shape[0], 1))
        # sensors1 = np.hstack([sensors1, zero_column])
        # change last column to zero
        s1[:, -1] = 0
        return np.hstack([s1, s2])


def plot_slit_sheet(tree: Slit, sheet: npt.NDArray):
    sh = make_rectangles_in_sheet(tree, deepcopy(sheet))
    # print(sheet)
    # print dataframe(sheet)
    # df = pd.DataFrame(sheet)
    # df.to_csv('output.csv', index=False)
    plt.imshow(sh)
    plt.show()


if __name__ == "__main__":
    example_tree = generate_random_slitting_tree(5)
    sheet = load_data("./data.csv")[0]
    # np.random.seed(1)
    # print(np.random.random((100, 100)))
    # for r in get_rectangles(example_tree, 100, 100, np.random.random((100, 100))):
    #     print(r.width, r.height)
    #     print(r.sensors)
    #     print()
    # sheet = load_data("./data.csv")[0]
    # plot_slits(example_tree, sheet)
    from simple_cut import Sheet

    sss = Sheet(sheet)
    sheet = sss.get_sheet()
    display_sheet(sheet)
    plot_slicing_tree(example_tree)
    # print(get_rectangles(example_tree, sheet))

    # print(average_weighted_worst_percentile(example_tree, sensors_sheet=sheet))
    # print(average_rectangle_size(example_tree, sensors_sheet=sheet))
    plot_slit_sheet(example_tree, sheet)
