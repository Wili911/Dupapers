import numpy as np

def global_contrast_normalization(X, s=1, lmda=10, epsilon=1e-8):
    X_average = np.mean(X)
    X = X - X_average
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)

    return X

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    sigma = np.cov(X) # [M x M]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 0.1 
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def zca_whitening(x):
    for i in range(x.shape[2]):
        # Before
        x[:,:,i] = np.dot(zca_whitening_matrix(x[:,:,i]), x[:,:,i])
    return x
