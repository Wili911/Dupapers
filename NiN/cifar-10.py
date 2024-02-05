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

from models import NiN
from preprocess import global_contrast_normalization, flatten, compute_zca_matrix
from DL import Trainer
from utils import seed_everything
from utils import set_device

# Set seed and device
seed_everything(0)
device = set_device()

# Set seed and device
seed_everything(0)
device = set_device()

# Load ZCA matrix
try:
    ZCA = np.load('cifar-10_zca.npy')
    print('ZCA matrix loaded')
except:
    print('Computing ZCA matrix')
    raw_train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=None)
    ZCA = compute_zca_matrix(raw_train_data.data[:,:,:,:])
    np.save('cifar-10_zca.npy', ZCA)

# Define the data transform
def custom_transform(x):
    x = np.array(x)
    x1 = x.reshape(-1)
    x2 = x1 - np.mean(x1)
    x3 = np.dot(x2, ZCA)
    x = np.reshape(x3, x.shape)
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    return x

# Load the data
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=custom_transform)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=custom_transform)

# Split train/validation data
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

# Instantiate model
model = NiN(lr=0.01).to(device)

# Initialize model
model.init(next(iter(train_dataloader))[0].to(device))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-6, lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

trainer = Trainer(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, device=device)

# Train model
trainer.train(epochs=50)

# Test model
test_loss, test_accuracy = trainer.test()
print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
