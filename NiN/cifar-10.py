import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from module.models import NiN
from module.preprocess import global_contrast_normalization, zca_whitening
from module.DL import Trainer
from module.utils import seed_everything
from module.utils import set_device



# Set seed and device
seed_everything(0)
device = set_device()

# Define the data transform
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

# Split train/validation data
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

# Instantiate model
model = NiN(lr=0.01).to(device)

# Initialize model
model.init(next(iter(train_dataloader))[0].to(device))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-6)

trainer = Trainer(model, loss_fn, optimizer, device=device)

# Train model
trainer.train(train_dataloader, val_dataloader, epochs=10)

# Test model
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

test_loss, test_accuracy = trainer.test(test_dataloader)
print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")