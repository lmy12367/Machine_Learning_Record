import matplotlib.image as mping
import matplotlib.pyplot as plt
import numpy as np

orig_img = np.array(mping.imread('./origimg.jpg'), dtype=int)
orig_img[orig_img < 128] = -1
orig_img[orig_img >= 128] = 1

noisy_img = np.array(mping.imread('noisyimg.jpg'), dtype=int)
noisy_img[noisy_img < 128] = -1
noisy_img[noisy_img >= 128] = 1

def compute_noise_rate(noisy,orig):
    err=np.sum(noisy!=orig)
    return err/orig.size

init_noise_rate=compute_noise_rate(noisy_img,orig_img)
print (f'带噪图像与原图不一致的像素比例：{init_noise_rate * 100:.4f}%')

def compute_energy(X,Y,i,j,alpha,beta):
    energy=-beta*X[i][j]*Y[i][j]

    if i>0:
        energy -=alpha*X[i][j]*X[i-1][j]
    if i<X.shape[0]-1:
        energy -= alpha * X[i][j] * X[i + 1][j]
    if j > 0:
            energy -= alpha * X[i][j] * X[i][j - 1]
    if j < X.shape[1] - 1:
            energy -= alpha * X[i][j] * X[i][j + 1]

    return energy

alpha=2.1
beta=1.0
max_iter=5

X=np.copy(noisy_img)

for k in range(max_iter):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j]=1
            pos_energy = compute_energy(X, noisy_img, i, j, alpha, beta)
            X[i, j] = -1
            neg_energy = compute_energy(X, noisy_img, i, j, alpha, beta)
            X[i, j] = 1 if pos_energy < neg_energy else -1

    plt.figure()
    plt.axis('off')
    plt.imshow(X, cmap='binary_r')
    plt.show()
    noise_rate = compute_noise_rate(X, orig_img) * 100
    print(f'迭代轮数：{k}，噪声率：{noise_rate:.4f}%')



