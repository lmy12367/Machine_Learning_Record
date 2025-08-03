from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_and_plot_data

x, y = load_and_plot_data('./data/ml/svm/spiral.csv')
model = SVC(kernel='rbf', gamma=50, tol=1e-6)
model.fit(x, y)

G = np.linspace(-1.5, 1.5, 100)
xx, yy = np.meshgrid(G, G)
X = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')
plt.legend()
plt.title("Sklearn SVM with RBF")
plt.show()