import numpy as np
from k_means import show_cluster
from init_methods import kmeanspp_init,random_init
from core import KMeans
if __name__=="__main__":
    data=np.array([[1,2],[2,3],[8,9],[9,10]])
    labels=np.array([0,0,1,1])
    show_cluster(data,labels,centroids=np.array([[1.5,2.5],[8.5,9.5]]),title="可视化")

    data = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
    K = 3

    print("=== random_init ===")
    centers1 = random_init(data, K, seed=42)

    print("\n=== kmeanspp_init ===")
    centers2 = kmeanspp_init(data, K, seed=42)

    data = np.vstack([
        np.random.randn(50, 2) + [2, 2],
        np.random.randn(50, 2) + [8, 8],
        np.random.randn(50, 2) + [2, 8]
    ])
    K = 3

    print("=== 随机初始化 ===")
    centroids1, labels1 = KMeans(data, K, random_init, max_iters=10, plot=False)

    print("\n=== k-means++ 初始化 ===")
    centroids2, labels2 = KMeans(data, K, kmeanspp_init, max_iters=10, plot=False)

    csv_path = "./data/ml/k_means/kmeans_data.csv"
    try:
        dataset = np.loadtxt(csv_path, delimiter=',')
    except OSError:
        print(f"[main] 找不到文件: {csv_path}")
        exit(1)

    print(f"[main] 成功加载数据，形状: {dataset.shape}")

    K = 4

    print("\n========== Random Initialization ==========")
    centroids_rand, labels_rand = KMeans(
        dataset, K, random_init, max_iters=20, plot=True
    )

    print("\n========== K-Means++ Initialization ==========")
    centroids_pp, labels_pp = KMeans(
        dataset, K, kmeanspp_init, max_iters=20, plot=True
    )

    print("\n[main] 最终结果对比：")
    print("随机初始化中心:\n", centroids_rand)
    print("k-means++ 中心:\n", centroids_pp)