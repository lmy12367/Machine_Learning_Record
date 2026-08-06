import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_and_plot_data
from smo import SMO 

if __name__=="__main__":
    x,y =load_and_plot_data("./data/ml/svm/linear.csv")
    C=1e8
    max_iter=1000
    np.random.seed(0)
    alpha, b = SMO(x, y, ker=np.inner, C=C, max_iter=max_iter)

    sup_idx = alpha > 1e-5
    print('支持向量个数：', np.sum(sup_idx))
    w = np.sum((alpha[sup_idx] * y[sup_idx]).reshape(-1, 1) * x[sup_idx], axis=0)
    print('参数 w 和 b：', w, b)

    X = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    Y = -(w[0] * X + b) / (w[1] + 1e-5)

    plt.figure()
    plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')
    plt.plot(X, Y, color='black')
    plt.scatter(x[sup_idx, 0], x[sup_idx, 1], marker='o', facecolors='none',
                edgecolors='purple', s=150, label='support vectors')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend()
    plt.title('Linear SVM with SMO')
    plt.show()