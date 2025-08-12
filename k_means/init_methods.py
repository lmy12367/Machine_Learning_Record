import numpy as np

def random_init(dataset,K,seed=None):
    if seed is not None:
        np.random.seed(seed)

    N=dataset.shape[0]
    print(f"数据集大小{N}，需要{K}个中心")

    idx=np.random.choice(np.arange(N),size=K,
                         replace=False)
    centers=dataset[idx]

    print(f"索引是{idx}")
    print(f"中心是{centers}")

    return centers

def kmeanspp_init(dataset,K,seed=None):
    if seed is not None:
        np.random.seed(seed)

    N,D=dataset.shape
    centers=[]

    first_idx=np.random.choice(N)
    centers.append(dataset[first_idx])
    print(f"第一个中心索引{first_idx},坐标{dataset[first_idx]}")


    for K in range(1,K):
        dist_sq = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in dataset])
        print(f"第{K}:距离平方{dist_sq}")

        probs=dist_sq/dist_sq.sum()
        next_idx=np.random.choice(N,p=probs)
        centers.append(dataset[next_idx])

        print(f"[kmeanspp_init] 第 {K+1} 个中心索引: {next_idx}, 坐标: {dataset[next_idx]}")

    return np.array(centers)

