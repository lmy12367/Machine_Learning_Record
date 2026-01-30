import math

def lp_distance(a, b, p):
    if p == float("inf"):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return (sum(abs(ai - bi) ** p for ai, bi in zip(a, b))) ** (1.0 / p)

# 3 个点：x1, x2, x3
x1 = (1, 1)
x2 = (5, 1)
x3 = (4, 4)

