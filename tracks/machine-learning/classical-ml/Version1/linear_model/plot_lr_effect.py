import numpy as np
import matplotlib.pyplot as plt
from sgd_manual import SGD  
from data_preprocess import x_train, y_train

num_epoch, batch_size = 20, 32
lrs = [0.1, 0.01, 0.001, 1.5]
logs = []
np.random.seed(0)

for lr in lrs:
    _, losses, _ = SGD(num_epoch, lr, batch_size)
    if lr == 1.5:                        
        plt.plot(np.log(losses), label=f'lr={lr}')
    else:
        plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epoch')
plt.ylabel('RMSE (log scale for lr=1.5)')
plt.legend()
plt.show()