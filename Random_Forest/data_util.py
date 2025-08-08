from tkinter import Radiobutton
from sklearn.datasets import make_classification,make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_STATE=42

def get_clf_data():
    x,y=make_classification(
        n_samples=1000,n_features=16,n_informative=5,
        n_redundant=2,n_classes=2,flip_y=0.1,
        random_state=RANDOM_STATE
    )

    return train_test_split(x,y,test_size=0.2,random_state=RANDOM_STATE)

def get_reg_data():
    x,y=make_friedman1(
        n_samples=2000,n_features=100,noise=0.5,
        random_state=RANDOM_STATE
    )

    return train_test_split(x,y,test_size=0.2,random_state=RANDOM_STATE)

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = get_clf_data()      
    print("分类数据集：")
    print("  训练集 shape:", X_tr.shape, y_tr.shape)
    print("  测试集 shape:", X_te.shape, y_te.shape)
    print("  训练集前 5 行特征：\n", X_tr[:5])
    print("  对应前 5 个标签：", y_tr[:5])

    print("\n" + "="*40 + "\n")

    X_tr, X_te, y_tr, y_te = get_reg_data()      
    print("回归数据集：")
    print("  训练集 shape:", X_tr.shape, y_tr.shape)
    print("  测试集 shape:", X_te.shape, y_te.shape)
    print("  训练集前 5 行特征：\n", X_tr[:5])
    print("  对应前 5 个标签：", y_tr[:5])
