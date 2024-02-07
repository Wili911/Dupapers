import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
DATA_DIR = ROOT / 'data/cifar-10'  # data directory

# Import custom modules
from module.models import NiN
from module.preprocess import global_contrast_normalization, flatten, compute_zca_transforms, ZCA_whitening
from module.DL import Trainer
from module.utils import seed_everything
from module.utils import set_device

# Set seed and device
seed_everything(0)
device = set_device()

# Load ZCA matrix
try:
    ZCA_matrix = torch.tensor(np.load(DATA_DIR / 'cifar-10_zca.npy')).float()
    print('ZCA matrix loaded')
except:
    print('Computing ZCA matrix.')
    raw_train_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    ZCA_matrix = compute_zca_transforms(raw_train_data.data[:,:,:,:], 0.1, save=True)
    

transform = transforms.Compose([
    ZCA_whitening(ZCA_matrix)
])

aug_transform = transforms.Compose([
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      ZCA_whitening(ZCA_matrix)
])

batch_size = 64

# Load the data
train_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)


    

# Split train/validation data
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

# Instantiate model
net = NiN().to(device)

# Initialize model
net.init(next(iter(train_loader))[0].to(device))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=2, min_lr=1e-6)

trainer = Trainer(net, loss_fn, optimizer, scheduler, train_loader, val_loader, test_loader, device=device, runs_dir='cifar-10_NiN')

# Train model
trainer.train(epochs=2)

# Test model
test_loss, test_accuracy = trainer.test()
print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
