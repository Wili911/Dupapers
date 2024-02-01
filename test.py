import torch
from torch.utils.data import DataLoader
from torch import nn

import torchvision

import random
import numpy as np
import os

from models import VGG
from models import NiN
from preprocess import global_contrast_normalization, zca_whitening
from DL_framework import Trainer

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
        else "mps"
    if torch.backends.mps.is_available()
        else "cpu"
)

print(f"Using {device} device")

# Set a seed to replicate results
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(0)

# Define the transformation of the train images.
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])

def custom_transform(x):
    x = np.array(x)
    x = global_contrast_normalization(x)
    x = zca_whitening(x)
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    return x

# Load the data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=custom_transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=custom_transform)

# train_data = torch.utils.data.Subset(train_data, np.random.choice(len(train_data), 10000, replace=False))
# test_data = torch.utils.data.Subset(test_data, np.random.choice(len(test_data), 1000, replace=False))

print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))

# Define dataloaders
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

# Data shape
print(train_data[0][0].shape)

# Instantiate model
#model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01).to(device)

model = NiN(lr=0.01).to(device)

# Initialize model
model.init(next(iter(train_dataloader))[0].to(device))
# print(model.net[0][0].weight[0])
# print(model.net[6].weight[0])

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(model, loss_fn, optimizer, device=device)

# Train model
trainer.train(train_dataloader, test_dataloader, epochs=10)

