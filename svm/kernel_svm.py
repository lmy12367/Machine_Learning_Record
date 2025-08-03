# kernel_svm.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from data_utils import load_and_plot_data
from SMO import SMO
from kernels import simple_poly_kernel, rbf_kernel, cos_kernel, sigmoid_kernel

if __name__ == "__main__":
    x, y = load_and_plot_data('./data/ml/svm/spiral.csv')
    kernels = [
        simple_poly_kernel(3),
        rbf_kernel(0.1),
        cos_kernel,
        sigmoid_kernel(1, -1)
    ]
    ker_names = ['Poly(3)', 'RBF(0.1)', 'Cos', 'Sigmoid(1,-1)']
    C = 100
    max_iter = 500
    np.random.seed(0)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    cmap = ListedColormap(['coral', 'royalblue'])

    for i, ker in enumerate(kernels):
        print('核函数：', ker_names[i])
        alpha, b = SMO(x, y, ker, C=C, max_iter=max_iter)
        sup_idx = alpha > 1e-6
        sup_x, sup_y, sup_alpha = x[sup_idx], y[sup_idx], alpha[sup_idx]

        def predict(x_new):
            s = 0
            for xi, yi, ai in zip(sup_x, sup_y, sup_alpha):
                s += yi * ai * ker(xi, x_new)
            return s + b

        G = np.linspace(-1.5, 1.5, 100)
        xx, yy = np.meshgrid(G, G)
        X = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([predict(xi) for xi in X])
        Z[Z < 0] = -1
        Z[Z >= 0] = 1
        Z = Z.reshape(xx.shape)

        axs[i].contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
        axs[i].scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')
        axs[i].scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')
        axs[i].set_title(ker_names[i])
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('kernel_svm_results.png')
    plt.show()