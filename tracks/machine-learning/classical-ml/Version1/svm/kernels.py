import numpy as np

def simple_poly_kernel(d):
    def k(x,y):
        return np.inner(x,y)**d
    return k

def rbf_kernel(sigma):
    def k(x,y):
        return np.exp(-np.inner(x-y,x-y)/(2.0*sigma**2))
    return k

def cos_kernel(x,y):
    return np.inner(x,y)/np.linalg.norm(x,2)/np.linalg.norm(y,2)

def sigmoid_kernel(beta,c):
    def k(x,y):
        return np.tanh(beta*np.inner(x,y)+c)
    return k