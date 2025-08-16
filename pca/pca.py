import numpy as np

def fit_transform(X,k):
    print("pca开始")
    X=X-np.mean(X,axis=0)
    cov=np.cov(X,rowvar=False)
    vals,vecs=np.linalg.eigh(cov)
    idx=np.argsort(vals)[::-1]
    W=vecs[:,idx[:k]]
    X_red=X@W
    print(f"降维后形状{X_red.shape}")

    return  X_red,W

if __name__ == "__main__":
    fake = np.random.randn(100, 4)
    Xr, _ = fit_transform(fake, k=2)