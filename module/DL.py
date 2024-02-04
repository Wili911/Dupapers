import torch
import torch.nn as nn

import inspect
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import os

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class DL_model(nn.Module, HyperParameters):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def init(self, dummy_input):
        self.forward(dummy_input)
        if self.init_weights is not None:
            self.net.apply(self.init_weights)

class Trainer():
    def __init__(self, model, loss, optimizer, train_dataloader=None, val_dataloader=None, test_dataloader=None, device='cpu'):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.writer = None
        self.train_dataloader = train_dataloader
        self.val_dataloader = None
        self.test_dataloader = test_dataloader
        self.epoch = 0

    def train_loop(self):
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        num_batches = len(self.train_dataloader)
        running_loss = 0.0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            # Compute prediction and loss
            pred = self.model(X)
            # Apply softmax to the output of the model
            loss = self.loss(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            N_log = 100
            if batch % N_log == (N_log - 1):
                running_loss /= N_log 
                # Log the running loss averaged per batch
                print(f"Epoch: {self.epoch}, Batch: {batch} / {num_batches}, Avg. Loss: {running_loss:.4f}")
                self.writer.add_scalar('loss/train',
                            running_loss,
                            self.epoch*num_batches + batch)
                running_loss = 0.0

    def val_loop(self):
        loss, accuracy = self.test(self.val_dataloader)
        self.writer.add_scalar('loss/test',
                            loss,
                            (self.epoch+1)*len(self.train_dataloader))
        self.writer.add_scalar('accuracy/test',
                            accuracy,
                            (self.epoch+1)*len(self.train_dataloader))
        print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_dataloader
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        num_batches = len(self.test_dataloader)
        loss, accuracy = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                # Compute prediction and loss
                pred = self.model(X)
                # Apply softmax to the output of the model
                loss += self.loss(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item() / len(y)
                
        loss /= num_batches
        accuracy /= num_batches
        return loss, accuracy
        
    def train(self, epochs=10, name=''):
        self.test_size = len(self.test_dataloader.dataset)
        self.train_size = len(self.train_dataloader.dataset)

        file_name = 'runs/'+name+'experiment_1'
        if os.path.exists(file_name):
            i = 2
            new_name = 'runs/'+name+'experiment_'+str(i)
            while os.path.exists(new_name):
                i += 1
                new_name = 'runs/'+name+'experiment_'+str(i)
            file_name = new_name
        self.writer = SummaryWriter(file_name)
        layout = {
            "ABCDE": {
                "loss": ["Multiline", ["loss/train", "loss/test"]],
                "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
            },
        }

        self.writer.add_custom_scalars(layout)

        for t in range(epochs):
            print(f"Epoch {t}\n-------------------------------")
            self.train_loop()
            self.test_loop()
            self.epoch += 1
        print("Done!")
        self.writer.flush()
        self.writer.close()

