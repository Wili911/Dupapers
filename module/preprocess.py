import numpy as np
import torch

def global_contrast_normalization(X):  
    X = X - np.mean(X, axis=0)
    X = X / np.sqrt((X ** 2).sum(axis=1))[:,None]
    return X

def flatten(X):
    return X.reshape(X.shape[0], -1)

def compute_zca_transforms(X, eps=1e-5, save=False):
    """ Compute ZCA whitening matrix with regularization parameter eps.
    Args: X: np.array [N, H, W, C]
        eps: float
        save: bool
    Returns: torch.tensor [D, D], D=HxWxC
    """
    X = flatten(X)
    X = (X - np.mean(X, axis=0)) / (np.sqrt((X ** 2).sum(axis=1))[:,None])
    s, u = np.linalg.eigh(X.T @ X)
    scale = np.sqrt(len(X) / (s+eps))
    ZCA_matrix = (u * scale) @ u.T 
    if save:
        np.save('cifar-10_zca.npy', ZCA_matrix)
    return torch.tensor(ZCA_matrix).float()

class ZCA_whitening(object):
    """Apply GCA + ZCA whitening to the input data.

    Args: ZCA_matrix: tensor [D,D], D=HxWxC
        mean_vector: tensor [D]
        variance: tensor [D]

    Returns: tensor [H, W, C]
    """

    def __init__(self, ZCA_matrix, eps=1e-5):
        assert isinstance(ZCA_matrix, torch.Tensor)
        D = ZCA_matrix.shape[0]
        self.D = D
        assert ZCA_matrix.shape == (D, D)

        self.ZCA_matrix = ZCA_matrix
        self.eps = eps
        self.D = D

    def __call__(self, pic):
        # Convert to tensor of shape (H, W, C)
        img = torch.as_tensor(np.array(pic, copy=True)).float()
        X = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        h, w, c = X.shape[:3]
        assert h * w * c == self.D

        # Flatten the image
        X = X.view(-1)

        # Apply GCA
        X = X - X.mean()
        X = X / torch.sqrt((X ** 2).sum())

        # Apply ZCA
        X = torch.matmul(X, self.ZCA_matrix)

        # Reshape the image
        X = X.view(h, w, c)
        X = X.permute(2, 0, 1)
        return X
        