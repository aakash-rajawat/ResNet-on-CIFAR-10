import os
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchsummary import summary

from ResNet56_implementation import *
from load_dataset import *


criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)

print(torch.cuda.is_available())


def train_model():
    EPOCHS = 15
    train_samples_num = 45000
    val_samples_num = 5000
    train_costs, val_costs = [], []

    # Training phase
    for epoch in range(EPOCHS):
        train_running_loss = 0
        correct_train = 0
        model.train().cuda()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()

            # Start the Forward Pass
            prediction = model(inputs)

            loss = criterion(prediction, labels)

            # Backpropagation
            loss.backward()
            optimiser.step()

            _, predicted_outputs = torch.max(prediction.data, 1)

            correct_train += (predicted_outputs == labels).float().sum().item()

            train_running_loss += (loss.data.item() * inputs.shape[0])

        train_epoch_loss = train_running_loss / train_samples_num

        train_costs.append(train_epoch_loss)

        train_acc = correct_train / train_samples_num

        val_running_loss = 0
        correct_val = 0

        model.eval().cuda()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                prediction = model(inputs)

                # Compute the loss
                loss = criterion(prediction, labels)

                # Compute the validation accuracy
                _, predicted_outputs = torch.max(prediction.data, 1)
                correct_val += (predicted_outputs == labels).float().sum().item()

            # Compute Batch loss
            val_running_loss += (loss.data.item() * inputs.shape[0])

            val_epoch_loss = val_running_loss / val_samples_num
            val_costs.append(val_epoch_loss)
            val_acc = correct_val / val_samples_num

        print(f"[Epoch: {epoch + 1}/{EPOCHS}]: train-loss = {train_epoch_loss:0.6f} | "
              f"train-acc = {train_acc:0.3f} | "
              f"val-loss = {val_epoch_loss:0.6f} | "
              f"val-acc = {val_acc:0.3f}")

        torch.save(model.state_dict(), "cache.pt")

    torch.save(model.state_dict(), "cache.pt")

    return train_costs, val_costs


train_costs, val_costs = train_model()

model = resnet56()
model.load_state_dict(torch.load("cache.pt"))
