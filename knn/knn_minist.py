import numpy as np
import matplotlib.pyplot as plt

m_x=np.loadtxt("./data/ml/knn_data/mnist_x",delimiter=' ')
m_y=np.loadtxt("./data/ml/knn_data/mnist_y")

# ---------- 2. 可视化第一张图 ----------
plt.figure()
plt.imshow(m_x[0].reshape(28, 28), cmap='gray')
plt.title(f'label = {int(m_y[0])}')
plt.show()


ratio = 0.8
split = int(len(m_x) * ratio)
np.random.seed(0)
idx = np.random.permutation(len(m_x))
m_x, m_y = m_x[idx], m_y[idx]
x_train, x_test = m_x[:split], m_x[split:]
y_train, y_test = m_y[:split], m_y[split:]


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class KNN:
    def __init__(self, k, label_num=10):
        self.k = k
        self.label_num = label_num

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_knn_indices(self, x):
        dis = list(map(lambda a: distance(a, x), self.x_train))
        knn_indices = np.argsort(dis)[:self.k]
        return knn_indices

    def get_label(self, x):
        knn_indices = self.get_knn_indices(x)
        label_stat = np.zeros(self.label_num)
        for idx in knn_indices:
            label = int(self.y_train[idx])
            label_stat[label] += 1
        return np.argmax(label_stat)

    def predict(self, x_test):
        pred = np.zeros(len(x_test), dtype=int)
        for i, x in enumerate(x_test):
            pred[i] = self.get_label(x)
        return pred


for k in range(1, 10):
    knn = KNN(k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    acc = np.mean(pred == y_test)
    print(f'K={k}, accuracy={acc*100:.1f}%')