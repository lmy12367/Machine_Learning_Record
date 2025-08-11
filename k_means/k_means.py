import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
group1=np.random.randn(50,2)+[2,2]
group2=np.random.randn(50,2)+[8,2]
group3=np.random.randn(50,2)+[2,8]
group4=np.random.randn(50,2)+[8,8]

dataset=np.vstack([group1,group2,group3,group4])
print(f"the number is {len(dataset)}")

def show_cluster(dataset,cluster,centroids=None):
    colors=np.array(['blue','red','green','purple'])
    plt.scatter(dataset[:,0],dataset[:,1],
               color=colors[cluster%len(colors)],
               alpha=0.6)
    if centroids is not None:
        k=len(centroids)
        plt.scatter(centroids[:,0],centroids[:,1],
                    color=colors[:K],
                    marker="X",edgecolors='black')
    plt.title("Clustering Result")
    plt.show()

def random_init(dataset,K):
    idx=np.random.choice(np.arange(len(dataset)),
                         size=K,
                         replace=False)
    return dataset[idx]

def Kmeans(dataset,K,init_cent):
    centroids=init_cent(dataset,K)
    cluster = np.zeros(len(dataset), dtype=int)
    changed = True
    itr = 0

if __name__=="__main__":
    show_cluster(dataset, np.zeros(len(dataset), dtype=int))
    centroids = random_init(dataset, 4)
    print("初始中心点：\n", centroids)