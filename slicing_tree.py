from dataclasses import dataclass
import random
from typing import Any
from anytree import NodeMixin


@dataclass
class Slit(NodeMixin):
    horizontal: bool
    offset: float  # percentage of the width/height

    def __repr__(self) -> str:
        return f"Slit(horizontal={self.horizontal}, offset={self.offset}, size={self.size})"


@dataclass
class Rectangle:
    width: float
    height: float
    sensors: Any


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

    if right_size > 0:
        children_right = (generate_random_slitting_tree(right_size),)
        children += children_right

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


def get_rectangles(node: Slit, width: float, height: float) -> list[Rectangle]:

    height1 = height * node.offset if node.horizontal else height
    height2 = height - height1 if node.horizontal else height

    width1 = width * node.offset if not node.horizontal else width
    width2 = width - width1 if not node.horizontal else width

    if not node.children:
        return [
            Rectangle(width1, height1, None),
            Rectangle(width2, height2, None),
        ]

    rectangles_children1 = get_rectangles(node.children[0], width1, height1)

    if len(node.children) == 1:
        return rectangles_children1

    rectangles_children2 = get_rectangles(node.children[1], width2, height2)

    return rectangles_children1 + rectangles_children2


def average_rectangle_size(tree: Slit, sheet_width: int, sheet_height: int) -> float:
    rectangles = get_rectangles(tree, sheet_width, sheet_height)
    return sum(rectangle.width * rectangle.height for rectangle in rectangles) / len(
        rectangles
    )
