import numpy as np
from data_preprocess import x_train, y_train, x_test, y_test

X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

thera=np.linalg.inv(X.T @ X) @ X.T @y_train
print("回归系数",thera)

y_pred = X_test@thera
rmse=np.sqrt(np.square(y_test-y_pred).mean())
print('RMSE：', rmse)

