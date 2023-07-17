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


class LambdaLayer(nn.Module):
    """
    This class defines a Lamda Layer. It allows to perform arbitrary operations
    specified by the "lambd" argument.
    Attributes:
        lambd: a function that defines the operation to be performed on the input.
    """
    def __init__(self, lambd):
        """
        Init method for the Lambda layer.
        :param lambd (function): Function that defines the operation to be performed
        on the input.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        """
        Forward pass of the Lambda layer. It applies the function to the input.
        :param x: Input tensor to the Lambda layer.
                  It is 4D tensor (number of samples in a batch, number of channels
                                   height, width)
        :return:
            torch.Tensor: The output of the Lamda layer after applying the function.
        """
        return self.lambd(x)


class BasicConvBlock(nn.Module):
    """
    The BasicConvBlock takes an input with in_channels, applies some blocks of
    convolutional layers to reduce it to out_channels and sum it up to the original
    input. If their sizes mismatch, then the input goes into an identity.
    Basically the BasicConvBlock will implement the regular basic Convolution Block
    + the shortcut block that does the dimension matching task (option A or B) when
    the dimension changes between 2 blocks.
    """
    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        """
        Init method for the Basic Convolution block.
        :param in_channels (int): Number of channels in the input tensor.
        :param out_channels (int): Number of channels in the output tensor.
        :param stride (int, optional): Stride for the convolution operation. Default is 1.
        :param option (str, optional): Option for the shortcut connection to match dimensions.
                                       Default is 'A'.
        """
        super(BasicConvBlock, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(out_channels)),
            ("act1", nn.ReLU()),
            ("bn2", nn.BatchNorm2d(out_channels))
        ]))

        self.shortcut = nn.Sequential()
        '''
            When input and output spatial dimensions don't match, we have 2 options
                A -> Use identity shortcuts with zero padding to increase channel 
                     dimension.
                B -> Use 1x1 convolution to increase channel dimension (projection
                     shortcut)
        '''
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                # Use identity shortcuts with zero padding to increase channel dimension
                pad_to_add = out_channels // 4
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],        # '::2' is doing a stride=2
                                                  (0, 0,                    # width beginning,width end ,
                                                   0, 0,                    # height beginning,height end ,
                                                   pad_to_add, pad_to_add,  # channel beginning, channel end
                                                   0, 0)))                  # Batch length beginning, end
                if option == 'B':
                    self.shortcut = nn.Sequential(OrderedDict([
                        "s_conv1", nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                        ("s_bn1", nn.BatchNorm2d(2 * out_channels))
                    ]))

    def forward(self, x):
        """
        Forward pass of the Basic Convolution block. It applies the sequence of
        layers and adds the shortcut connection.
        :param x:  Intput torch.Tensor to the Basic Convolution block.
        :return:
            torch.Tensor: The output of the Basic Convolution block.
        """
        out = self.features(x)
        out += self.shortcut(x)  # Sum up the shortcut layer
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet 56 architecture for CIFAR-10 dataset of shape 32*32*3.
    Arguments:
        block_type (nn.Module): The type of residual block to use.
        num_blocks (list):      The list containing the number of blocks for each layer.
    Attributes:
        in_channels (int):      Number of input channels.
        conv0 (nn.Conv2d):      Initial convolutional layer.
        bn0 (nn.BatchNorm2d):   Batch normalization layer.
        block1 (nn.Sequential): First block layer.
        block2 (nn.Sequential): Second block layer.
        block3 (nn.Sequential): Third block layer.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        linear (nn.Linear):     Linear layer for classification.
    """
    def __init__(self, block_type, num_blocks):
        super(ResNet, self).__init__()

        self.in_channels = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.block1 = self.__build_layer(block_type, 16, num_blocks[0], starting_stride=1)

        self.block2 = self.__build_layer(block_type, 32, num_blocks[1], starting_stride=2)

        self.block3 = self.__build_layer(block_type, 64, num_blocks[2], starting_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(64, 10)

    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        """
        Build a layer consisting of multiple residual blocks.
        :param block_type (nn.Module): The type of residual block to use.
        :param out_channels (int): Number of output channels.
        :param num_blocks (int): Number of blocks in the layer.
        :param starting_stride (int): Stride value for the first block.
        :return:
            nn.Sequential: Sequential container of residual blocks.
        """
        # Generating an array whose first element is starting_stride
        # and it will have (num_blocks - 1) more elements each of value 1.
        strides_list_for_current_block = [starting_stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides_list_for_current_block:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.
        :param x: Input torch.Tensor
        :return:
            torch.Tensor: Output
        """
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def resnet56():
    return ResNet(block_type=BasicConvBlock, num_blocks=[9, 9, 9])


model = resnet56()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model.to(device)
summary(model, (3, 32, 32))
