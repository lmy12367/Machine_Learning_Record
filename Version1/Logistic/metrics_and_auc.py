from re import I
import numpy as np
def acc(y_true,y_pred):
    return np.mean(y_true == y_pred)

def auc(y_true, y_pred):
    idx=np.argsort(y_pred)[::-1]
    y_true=y_true[idx]

    tp=np.cumsum(y_true)
    fp=np.cumsum(1-y_true)
    tpr = np.concatenate([[0], tp / tp[-1]])
    fpr = np.concatenate([[0], fp / fp[-1]])
    s=0.0
    for i in range(1,len(fpr)):
        s += (fpr[i]-fpr[i-1])*tpr[i]
    return s

if __name__ == '__main__':
    y_true = np.array([1, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.3])
    print('ACC =', acc(y_true, y_prob >= 0.5))
    print('AUC =', auc(y_true, y_prob))
    
    