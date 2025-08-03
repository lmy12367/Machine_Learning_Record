import numpy as np
from tqdm import trange
import matplotlib as plt

def SMO(x, y, ker, C, max_iter):
    m = x.shape[0]
    alpha = np.zeros(m)
    b = 0
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = ker(x[i], x[j])

    for _ in trange(max_iter):
        for i in range(m):
            j = np.random.choice([l for l in range(m) if l != i])
            eta = K[j, j] + K[i, i] - 2 * K[i, j]
            e_i = np.sum(y * alpha * K[:, i]) + b - y[i]
            e_j = np.sum(y * alpha * K[:, j]) + b - y[j]

            alpha_i_old = alpha[i]
            alpha_j_old = alpha[j]

            alpha_i = alpha_i_old + y[i] * (e_j - e_i) / (eta + 1e-5)
            zeta = alpha_i_old * y[i] + alpha_j_old * y[j]

            if y[i] == y[j]:
                lower = max(0, zeta / y[i] - C)
                upper = min(C, zeta / y[i])
            else:
                lower = max(0, zeta / y[i])
                upper = min(C, zeta / y[i] + C)

            alpha_i = np.clip(alpha_i, lower, upper)
            alpha_j = (zeta - y[i] * alpha_i) / y[j]

            b_i = b - e_i - y[i] * (alpha_i - alpha_i_old) * K[i, i] - y[j] * (alpha_j - alpha_j_old) * K[i, j]
            b_j = b - e_j - y[j] * (alpha_j - alpha_j_old) * K[j, j] - y[i] * (alpha_i - alpha_i_old) * K[i, j]

            if 0 < alpha_i < C:
                b = b_i
            elif 0 < alpha_j < C:
                b = b_j
            else:
                b = (b_i + b_j) / 2

            alpha[i], alpha[j] = alpha_i, alpha_j

    return alpha, b