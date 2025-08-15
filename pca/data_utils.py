import pandas as pd
import numpy as np

def load_numeric_csv(path: str, usecols=None) -> np.ndarray:
    """读取 CSV 并返回纯数值 ndarray"""
    print(f"[data_utils] 读取 {path}")
    df = pd.read_csv(path, usecols=usecols)
    arr = df.values.astype(float)
    print(f"[data_utils] 数据形状 {arr.shape}，前 3 行：\n{arr[:3]}")
    return arr

if __name__ == "__main__":
    # 直接运行本文件即可验证
    _ = load_numeric_csv("my_data.csv")