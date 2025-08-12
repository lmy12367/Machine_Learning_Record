import matplotlib.pyplot as plt
import numpy as np

def show_cluster(dataset,cluster,centroids=None,title="Clustering Result"):
    print(f"数据形状{dataset.shape},聚类标签长度{len(cluster)}")
    
    if centroids is not None:
        print(f"聚类中心形状{centroids.shape}")
    else:
        print("没有数据，不绘制中心点")

    colors=np.array(["blue","red","green","purple"])
    plt.figure()
    plt.scatter(dataset[:,0],dataset[:,1],
                c=colors[cluster],s=30)
    
    if centroids is not None:
        plt.scatter(centroids[:,0],centroids[:,1],
                    c=colors[:len(centroids)],marker="+",s=200)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

