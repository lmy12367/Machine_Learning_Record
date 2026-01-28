import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from metrics_and_auc import acc,auc

data = np.load('./data/ml/Logistic/split.npz')
x_train, y_train = data['x_train'], data['y_train']
x_test,  y_test  = data['x_test'],  data['y_test']

X = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

def logistic(z):
    return 1/(1+np.exp(-z))

def GD(num_steps, lr, l2_coef):
    theta = np.random.normal(size=X.shape[1])
    train_losses, test_losses = [], []
    train_accs,  test_accs  = [], []
    train_aucs,  test_aucs  = [], []

    for _ in range(num_steps):
        
        pred = logistic(X @ theta)
        grad = -X.T @ (y_train - pred) + l2_coef * theta
        theta -= lr * grad

    
        train_loss = (- y_train @ np.log(pred + 1e-12)
                      - (1 - y_train) @ np.log(1 - pred + 1e-12)
                      + l2_coef * np.dot(theta, theta) / 2) / len(X)
        train_losses.append(train_loss)
        train_accs.append(acc(y_train, pred >= 0.5))
        train_aucs.append(auc(y_train, pred))

        
        test_pred = logistic(X_test @ theta)
        test_loss = (- y_test @ np.log(test_pred + 1e-12)
                     - (1 - y_test) @ np.log(1 - test_pred + 1e-12)) / len(X_test)
        test_losses.append(test_loss)
        test_accs.append(acc(y_test, test_pred >= 0.5))
        test_aucs.append(auc(y_test, test_pred))

    return theta, train_losses, test_losses, train_accs, test_accs, train_aucs, test_aucs
    


np.random.seed(0)
theta, *rets = GD(num_steps=250, lr=0.002, l2_coef=1.0)
train_losses, test_losses, train_accs, test_accs, train_aucs, test_aucs = rets


y_pred = (logistic(X_test @ theta) >= 0.5).astype(int)
print('预测准确率：', acc(y_test, y_pred))
print('回归系数：', theta)


plt.figure(figsize=(13, 9))
x = np.arange(1, 251)
plt.subplot(221); plt.plot(x, train_losses, label='train loss')
plt.plot(x, test_losses,  '--', label='test loss'); plt.legend()
plt.subplot(222); plt.plot(x, train_accs, label='train acc')
plt.plot(x, test_accs,  '--', label='test acc');  plt.legend()
plt.subplot(223); plt.plot(x, train_aucs, label='train auc')
plt.plot(x, test_aucs,  '--', label='test auc');  plt.legend()


plt.subplot(224)
pos = np.load('./data/ml/Logistic/split.npz')['x_total'][np.load('./data/ml/Logistic/split.npz')['y_total'] == 1]
neg = np.load('./data/ml/Logistic/split.npz')['x_total'][np.load('./data/ml/Logistic/split.npz')['y_total'] == 0]
plt.scatter(pos[:, 0], pos[:, 1], marker='o', color='coral', s=10, label='positive')
plt.scatter(neg[:, 0], neg[:, 1], marker='x', color='blue',  s=10, label='negative')
plot_x = np.linspace(-1.1, 1.1, 100)
plot_y = -(theta[0] * plot_x + theta[2]) / theta[1]
plt.plot(plot_x, plot_y, ls='-.', color='green', label='decision boundary')
plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
plt.xlabel('X1'); plt.ylabel('X2'); plt.legend()
plt.tight_layout()
plt.show()