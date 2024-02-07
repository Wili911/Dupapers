import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    return device

def show(X):
    # Plot image by bringing the pixel values between 0 and 1
    X = X - X.min()
    X = X / X.max()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(X)
    plt.show()