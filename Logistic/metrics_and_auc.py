import numpy as np
def acc(y_true,y_pred):
    return np.mean(y_true == y_pred)

def auc(y_true, y_pred):
    idx=np.argsort(y_pred)[::-1]
    