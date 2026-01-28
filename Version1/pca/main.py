import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_numeric_csv(path: str, usecols=None) -> np.ndarray:
    print(f"[data_utils] 读取 {path}")
    df = pd.read_csv(path, usecols=usecols)
    arr = df.values.astype(float)
    print(f"[data_utils] 数据形状 {arr.shape}，前 3 行：\n{arr[:3]}")
    return arr

def pca_manual(X: np.ndarray, k: int):
    print("[pca_manual] 手写 PCA 开始")
    X = X - np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    W = vecs[:, idx[:k]]
    X_red = X @ W
    print(f"[pca_manual] 降维后形状 {X_red.shape}")
    return X_red, W

def pca_sklearn(X: np.ndarray, k: int):
    print("[pca_sklearn] sklearn PCA 开始")
    model = PCA(n_components=k)
    X_red = model.fit_transform(X)
    print(f"[pca_sklearn] 降维后形状 {X_red.shape}")
    return X_red, model

# ---------- main ----------
def main():
    X = load_numeric_csv("./data/PCA_dataset.csv")

    X1, _ = pca_manual(X, k=2)

    X2, _ = pca_sklearn(X, k=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X1[:, 0], X1[:, 1], s=10)
    plt.title("手写 PCA")
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(X2[:, 0], X2[:, 1], s=10, c='green')
    plt.title("sklearn PCA")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()