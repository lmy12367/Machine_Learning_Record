import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data=np.loadtxt("./data/ml/Bilinear_model/movielens_100k.csv",delimiter=',',dtype=int)
print("data len",len(data))

data[:,:2]=data[:,:2]-1

user =set()
items=set()

for i,j,k in data:
    user.add(i)
    items.add(j)

use_num=len(user)
item_num=len(items)

print(f"user number,{use_num},movie number is {item_num}")

np.random.seed(0)

ratio=0.8
spilt=int(len(data)*ratio)
np.random.shuffle(data)
train=data[:spilt]
test=data[spilt:]

user_cnt = np.bincount(train[:, 0], minlength=use_num)
item_cnt = np.bincount(train[:, 1], minlength=item_num)
print(user_cnt[:10])
print(item_cnt[:10])

user_train,user_test = train[:, 0], test[:, 0]
item_train,item_test = train[:, 1], test[:, 1]
y_train,y_test = train[:, 2], test[:, 2]

class MF:
    def __init__(self,N,M,d):
        self.user_params=np.ones((N,d))
        self.item_params=np.ones((M,d))

    def pred(self,user_id,item_id):
        user_params=self.user_params[user_id]
        item_params=self.item_params[item_id]

        rating_pred=np.sum(user_params*item_params,axis=1)
        return rating_pred

    def update(self,user_grad,item_grad,lr):
        self.user_params -= lr*user_grad
        self.item_params -= lr*item_grad

def train(model,learning_rate,lbd,max_training_step,batch_size):
    train_losses=[]
    test_loss=[]
    batch_num=int(np.ceil(len(user_train)/batch_size))
    with tqdm(range(max_training_step*batch_num)) as pbar:
        for epoch in range(max_training_step):
            train_rmse=0
            for i in range(batch_num):
                st=i *batch_size
                ed = min(len(user_train), st + batch_size)
                user_batch = user_train[st: ed]
                item_batch = item_train[st: ed]
                y_batch = y_train[st: ed]

                y_pred = model.pred(user_batch, item_batch)
                
                P = model.user_params
                Q = model.item_params

                errs = y_batch - y_pred
                P_grad = np.zeros_like(P)
                Q_grad = np.zeros_like(Q)

                for user, item, err in zip(user_batch, item_batch, errs):
                    P_grad[user] = P_grad[user] - err * Q[item] + lbd * P[user]
                    Q_grad[item] = Q_grad[item] - err * P[user] + lbd * Q[item]
                
                model.update(P_grad / len(user_batch), Q_grad / len(user_batch), learning_rate)

                train_rmse += np.mean(errs ** 2)

                pbar.set_postfix({
                    'Epoch': epoch,
                    'Train RMSE': f'{np.sqrt(train_rmse / (i + 1)):.4f}',
                    'Test RMSE': f'{test_loss[-1]:.4f}' if test_loss else None
                })
                pbar.update(1)
            
            
            train_rmse = np.sqrt(train_rmse / len(user_train))
            train_losses.append(train_rmse)
            y_test_pred = model.pred(user_test, item_test)
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            test_loss.append(test_rmse)

    return train_losses, test_loss

if __name__ == '__main__':
    feature_num = 16 
    learning_rate = 0.1 
    lbd = 1e-4 
    max_training_step = 30
    batch_size = 64 


    model = MF(use_num, item_num, feature_num)

    train_losses, test_losses = train(model, learning_rate, lbd,
    max_training_step, batch_size)

    plt.figure()
    x = np.arange(max_training_step) + 1
    plt.plot(x, train_losses, color='blue', label='train loss')
    plt.plot(x, test_losses, color='red', ls='--', label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

        
