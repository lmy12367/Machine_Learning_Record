import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[float, ...]  # k 维点

def l2(a: Point, b: Point) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

@dataclass
class KDNode:
    point: Point
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None