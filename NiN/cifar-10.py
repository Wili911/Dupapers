import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(Path(os.path.dirname( __file__ )).parent, 'module')) 

from models import NiN, VGG
from preprocess import global_contrast_normalization, flatten, compute_zca_matrix
from DL import Trainer
from utils import seed_everything
from utils import set_device

# Set seed and device
seed_everything(0)
device = set_device()

# Load ZCA matrix
# try:
#     ZCA = np.load('cifar-10_zca.npy')
#     print('ZCA matrix loaded')
# except:
#     print('Computing ZCA matrix')
#     raw_train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=None)
#     ZCA = compute_zca_matrix(raw_train_data.data[:,:,:,:],0.1)
#     np.save('cifar-10_zca.npy', ZCA)

# Define the data transform
# def custom_transform(x):
#     x = np.array(x)
#     x1 = x.reshape(-1)
#     x1 = x1 - np.mean(x1)
#     x1 = x1 / np.sqrt((x1 ** 2).sum())
#     x1 = np.dot(x1, ZCA)
#     x = np.reshape(x1, x.shape)
#     x = torch.from_numpy(x).float()
#     x = x.permute(2, 0, 1)
#     return x

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


batch_size = 64

# Load the data
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

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
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-6, lr=0.001)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=2, min_lr=1e-6)

# scheduler = None
trainer = Trainer(net, loss_fn, optimizer, scheduler, train_loader, val_loader, test_loader, device=device)

# Train model
trainer.train(epochs=2)

# Test model
test_loss, test_accuracy = trainer.test()
print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
