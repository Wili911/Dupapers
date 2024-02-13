import os
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from module.models import nin_block
from module.models import init_weights
from module.DL import Trainer
from module.DL import DL_model
from module.utils import seed_everything
from module.utils import set_device
from module.config import MNIST_DIR



# Set seed and device
seed_everything(0)
device = set_device()


# Load raw data
raw_train_data = torchvision.datasets.MNIST(root=MNIST_DIR, train=True, download=True, transform=None)

# Define the data transform
custom_transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((raw_train_data.data.float().mean() / 255,), (raw_train_data.data.float().std() / 255))])

# Load the data with the custom transform
train_data = torchvision.datasets.MNIST(root=MNIST_DIR, train=True, download=True, transform=custom_transform)
test_data = torchvision.datasets.MNIST(root=MNIST_DIR, train=False, download=True, transform=custom_transform)


# Split train/validation data
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

class NiN(DL_model):
    def __init__(self, lr=0.1, num_classes=10, init_weights=init_weights):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(32, kernel_size=7, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(64, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.3),
            nin_block(num_classes, kernel_size=3, strides=1, padding=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
    
# Instantiate model
model = NiN(lr=0.01).to(device)

# Initialize model
model.init(next(iter(train_dataloader))[0].to(device))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader, test_dataloader, device=device)

# Train model
trainer.train(epochs=50)

# Test model
test_loss, test_accuracy = trainer.test(test_dataloader)
print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")