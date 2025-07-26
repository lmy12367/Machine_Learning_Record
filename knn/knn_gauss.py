from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("./data/ml/knn_data/gauss.csv",delimiter=",")

x_train = data[ : , :2]
y_train = data[ : ,2]

print(len(x_train))

plt.figure()
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],
            c="blue",marker='o',label='class 0')
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],
            c='red',marker='x',label='class 1')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

step=0.02
x_min,x_max = x_train[:,0].min()-1,x_train[:,0].max()+1
y_min,y_max = x_train[:,1].min()-1,x_train[:,1].max()+1

xx,yy=np.meshgrid(np.arange(x_min,x_max,step),
                  np.arange(y_min,y_max,step))

grid = np.c_[xx.ravel(),yy.ravel()]

fig = plt.figure(figsize=(16,4.5))
cmap_light=ListedColormap(['royalblue', 'lightcoral'])

for i ,k in enumerate([1,3,10]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)

    z=knn.predict(grid).reshape(xx.shape)
    ax=fig.add_subplot(1,3,i+1)
    ax.pcolormesh(xx, yy, z, cmap=cmap_light, alpha=0.7)
    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1],
               c='blue', marker='o', edgecolors='k')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1],
               c='red', marker='x', edgecolors='k')
    ax.set_title(f'K = {k}')
plt.show()

