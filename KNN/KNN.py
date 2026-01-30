import math
from collections import Counter

def lp_distance(a, b, p=2):
    """Lp 距离：对应式(3.2)；p=2/1/inf 对应式(3.3)/(3.4)/(3.5)"""
    if p == float("inf"):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return (sum(abs(ai - bi) ** p for ai, bi in zip(a, b))) ** (1.0 / p)



