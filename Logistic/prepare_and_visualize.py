import numpy as np
import matplotlib.pyplot as plt


lines = np.loadtxt('data/lr_dataset.csv', delimiter=',', dtype=float)
x_total = lines[:, 0:2]
y_total = lines[:, 2]
print('数据集大小：', len(x_total))

pos_idx = np.where(y_total == 1)
neg_idx = np.where(y_total == 0)
plt.scatter(x_total[pos_idx, 0], x_total[pos_idx, 1],
            marker='o', color='coral', s=10, label='positive')
plt.scatter(x_total[neg_idx, 0], x_total[neg_idx, 1],
            marker='x', color='blue', s=10, label='negative')
plt.xlabel('X1 axis'); plt.ylabel('X2 axis')
plt.legend(); plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300)
plt.show()

np.random.seed(0)
ratio = 0.7
split = int(len(x_total) * ratio)
idx = np.random.permutation(len(x_total))
x_total, y_total = x_total[idx], y_total[idx]
x_train, y_train = x_total[:split], y_total[:split]
x_test,  y_test  = x_total[split:], y_total[split:]


np.savez('data/split.npz',
         x_train=x_train, y_train=y_train,
         x_test=x_test,   y_test=y_test)