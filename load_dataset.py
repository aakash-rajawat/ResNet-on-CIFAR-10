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


def dataloader_cifar_10():
    """
    Create dataloaders for the CIFAR-10 dataset.
    :return:
        train_loader (torch.utils.data.DataLoader): Dataloader for the training set.
        val_loader (torch.utils.data.DataLoader): Dataloader for the validation set.
        test_loader (torch.utils.data.DataLoader): Dataloader for the test set.
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = datasets.CIFAR10("../input_data", train=True, download=True, transform=transform)

    test_dataset = datasets.CIFAR10("../input_data", train=False, download=True, transform=transform)

    # Splitting training data into training and validation dataset
    train_dataset, val_dataset = random_split(train_dataset, (45000, 5000))

    if __name__ == "__main__":
        print(f"Image shape of a random sample image: {train_dataset[0][0].numpy().shape}", end="\n\n")
        print(f"Training dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
        print(f"Test dataset: {len(test_dataset)} images")

    BATCH_SIZE = 32

    # Generating the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = dataloader_cifar_10()
