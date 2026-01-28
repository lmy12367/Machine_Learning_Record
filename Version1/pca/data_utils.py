import numpy as np
import pandas as pd


def load_numeric_csv(path,usecols=None):
    print(f"读取{path}")
    df=pd.read_csv(path,usecols=usecols)
    arr=df.values.astype(float)
    print(f"数据形状{arr.shape},前三行数据{arr[:3]}")

    return arr

if __name__=="__main__":
    _=load_numeric_csv('./data/PCA_dataset.csv')