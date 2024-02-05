import numpy as np

def global_contrast_normalization(X, s=1, lmda=10, epsilon=1e-8):
    X_average = np.mean(X)
    X = X - X_average
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)

    return X

def zca_whitening_matrix(X, epsilon=1e-5):
    sigma = np.cov(X)
    U,S,V = np.linalg.svd(sigma)
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix

def zca_whitening(x):
    for i in range(x.shape[2]):
        x[:,:,i] = np.dot(zca_whitening_matrix(x[:,:,i]), x[:,:,i])
    return x

def zca_matrix(images_tensor):
    # Center the data
    mean = np.mean(images_tensor, axis=(1, 2, 3))
    centered_data = images_tensor - mean

    # Flatten the data
    flat_data = centered_data.reshape(-1, np.prod(images_tensor.shape[1:3].numpy()))
    print(flat_data.shape)
    