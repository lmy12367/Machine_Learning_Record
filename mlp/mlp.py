from re import X
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('./data/ml/mlp/xor_dataset.csv',delimiter=",")
print(len(data))
print(data[:5])

ratio=0.8
split=int(ratio*len(data))

np.random.seed(0)

data=np.random.permutation(data)
x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, learning_rate):
        pass

class Linear(Layer):
    def __init__(self,num_in,num_out,use_bias=True):
        super().__init__()
        self.num_in=num_in
        self.num_out=num_out
        self.use_bias=use_bias

        self.W=np.random.normal(loc=0,scale=1.0,size=(num_in,num_out))
        if use_bias: 
            self.b = np.zeros((1, num_out))