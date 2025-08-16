from sklearn.decomposition import PCA
import numpy as np

def fit_transform(X,k):
    print("sklearn PCA")
    model=PCA(n_components=k)
    X_red=model.fit_transform(X)
    print(f"降维之后的形状{X_red.shape}")

    return  X_red,model

if __name__ == "__main__":
    fake = np.random.randn(100, 4)
    Xr, _ = fit_transform(fake, k=2)