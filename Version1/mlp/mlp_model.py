import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./data/ml/mlp/xor_dataset.csv', delimiter=",")
print(len(data))
print(data[:5])

ratio = 0.8
split = int(ratio * len(data))

np.random.seed(0)
data = np.random.permutation(data)
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
    def __init__(self, num_in, num_out, use_bias=True):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_bias = use_bias

        self.W = np.random.normal(loc=0, scale=1.0, size=(num_in, num_out))
        if use_bias: 
            self.b = np.zeros((1, num_out))

    def forward(self, x):
        self.x = x 
        self.y = x @ self.W
        
        if self.use_bias:
            self.y += self.b
        
        return self.y
    
    def backward(self, grad):
        self.grad_W = self.x.T @ grad / grad.shape[0]
        
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        
        grad = grad @ self.W.T
        return grad
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b

class Identity(Layer):
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

class Sigmoid(Layer):
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)

class Tanh(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1 - self.y ** 2)

class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    def backward(self, grad):
        return grad * (self.x >= 0)

activation_dict = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU
}

class MLP:
    def __init__(self, layer_sizes, use_bias=True, activation='relu', out_activation='identity'):
        self.layers = []
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1:-1]:
            self.layers.append(Linear(num_in, num_out, use_bias))
            self.layers.append(activation_dict[activation]())
            num_in = num_out

        self.layers.append(Linear(num_in, layer_sizes[-1], use_bias))
        self.layers.append(activation_dict[out_activation]())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

if __name__ == "__main__":
    num_epochs = 1000
    learning_rate = 0.1
    batch_size = 128
    eps = 1e-7 

    mlp = MLP(layer_sizes=[2, 4, 1], use_bias=True, out_activation='sigmoid')

    losses = []
    test_losses = []
    test_accs = []
    for epoch in range(num_epochs):
        st = 0
        loss = 0.0
        while True:
            ed = min(st + batch_size, len(x_train))
            if st >= ed:
                break
            
            x = x_train[st:ed]
            y = y_train[st:ed]
            
            y_pred = mlp.forward(x)
            
            grad = y_pred - y
            
            mlp.backward(grad)
            mlp.update(learning_rate)
            
            train_loss = np.sum(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))
            loss += train_loss
            st += batch_size

        losses.append(loss / len(x_train))
    
        y_pred = mlp.forward(x_test)
        test_loss = np.sum(-y_test * np.log(y_pred + eps) - (1 - y_test) * np.log(1 - y_pred + eps)) / len(x_test)
        test_acc = np.sum(np.round(y_pred) == y_test) / len(x_test)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print('测试精度：', test_accs[-1])

    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(losses, color='blue', label='train loss')
    plt.plot(test_losses, color='red', ls='--', label='test loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Cross-Entropy Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(test_accs, color='red')
    plt.ylim(top=1.0)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.show()
