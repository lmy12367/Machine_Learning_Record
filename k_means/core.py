import numpy as np
from k_means import show_cluster

def KMeans(dataset,K,init_func,max_iters=100,tol=1e-4,plot=True):
    N,D=dataset.shape
    print(f"[KMeans] 数据集大小: {N}，维度: {D}，聚类数: {K}")

    centroids=init_func(dataset,K)
    print(f"[KMeans] 初始中心:\n{centroids}")

    cluster = np.zeros(N, dtype=int)
    for it in range(max_iters):
        print(f"\n--- Iteration {it} ---")

        distances = np.linalg.norm(dataset[:, None] - centroids[None, :], axis=2)  # (N, K)
        new_cluster = np.argmin(distances, axis=1)

        print(f"[KMeans] 类别分布: {np.bincount(new_cluster)}")
    
        new_centroids = np.array([
            dataset[new_cluster == k].mean(axis=0) if np.any(new_cluster == k) else centroids[k]
            for k in range(K)
        ])

        print(f"[KMeans] 新中心:\n{new_centroids}")


        shift = np.linalg.norm(new_centroids - centroids)
        print(f"[KMeans] 中心移动距离: {shift:.5f}")

        if plot:
            show_cluster(dataset, new_cluster, new_centroids, title=f"Iteration {it}")

        if shift < tol:
            print(f"[KMeans] 收敛于第 {it} 轮")
            break

        centroids = new_centroids
        cluster = new_cluster

    return centroids, cluster