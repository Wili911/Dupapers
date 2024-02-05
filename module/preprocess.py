import numpy as np

def global_contrast_normalization(X):  
    X = X - np.mean(X, axis=0)
    X = X / np.sqrt((X ** 2).sum(axis=1))[:,None]
    return X

def flatten(X):
    return X.reshape(X.shape[0], -1)

def compute_zca_matrix(X):
    X = flatten(X)
    X = global_contrast_normalization(X)
    s, u = np.linalg.eigh(X.T @ X)
    scale = np.sqrt(len(X) / s)
    return (u * scale) @ u.T
