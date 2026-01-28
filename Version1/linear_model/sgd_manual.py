import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from data_preprocess import x_train, y_train, x_test, y_test


def batch_generator(x, y, batch_size, shuffle=True):
    batch_count = 0
    if shuffle:
        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]
    while True:
        start = batch_count * batch_size
        end   = min(start + batch_size, len(x))
        if start >= end:
            break
        batch_count += 1
        yield x[start:end], y[start:end]


def SGD(num_epoch, learning_rate, batch_size):
    X = np.concatenate([x_train, 
                        np.ones((len(x_train), 1))],
                        axis=-1)
    
    X_test = np.concatenate([x_test,  
                            np.ones((len(x_test), 1))], 
                            axis=-1)
    
    theta = np.random.normal(size=X.shape[1])

    train_losses, test_losses = [], []
    
    for _ in range(num_epoch):
        gen = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0
        for xb, yb in gen:
            grad = xb.T @ (xb @ theta - yb)
            theta -= learning_rate * grad / len(xb)
            train_loss += np.square(xb @ theta - yb).sum()
        train_losses.append(np.sqrt(train_loss / len(X)))
        test_losses.append(np.sqrt(np.square(X_test @ theta - y_test).mean()))
    print('回归系数：', theta)
    return theta, train_losses, test_losses

if __name__ == '__main__':
    num_epoch, lr, bs = 20, 0.01, 32
    np.random.seed(0)
    _, tr, te = SGD(num_epoch, lr, bs)

    plt.plot(tr, label='train loss')
    plt.plot(te, '--', label='test loss')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.legend(); plt.show()