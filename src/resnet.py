import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
torch.set_default_dtype(torch.float64)

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock2, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features)  # 直接线性变换，无 BatchNorm
            )

    def forward(self, x):
        out = F.relu(self.fc1(x))  # 直接 ReLU，无 BatchNorm
        out = self.fc2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_dim, dictionary_dim=128):
        super(ResNet, self).__init__()
        self.in_features = input_dim
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2])
        self.linear = nn.Linear(64, dictionary_dim)

    def _make_layer(self, block, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, out_features))
            self.in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.linear(out)
        return out
    
    def initialize_weights_to_zero(self):
        """ 将所有参数初始化为 0 """
        for param in self.parameters():
            nn.init.constant_(param, 0)
class TrainableDictionary(nn.Module):
    """
    A neural network module that builds a trainable dictionary for input data.
    Args:
        inputs_dim (int): The dimension of the input data.
        dictionary_dim (int): The dimension of the dictionary output.
        resnet_params (tuple): Parameters to define the ResNet structure.
    Methods:
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x` and returns the concatenated tensor
            of ones, the input `x`, and the dictionary output.
    """

    def __init__(self, input_dim, dictionary_dim, num_blocks):
        super(TrainableDictionary, self).__init__()
        self.input_dim = input_dim
        self.dictionary_dim = dictionary_dim
        
        # Initialize the ResNet model
        self.resnet = ResNet(
            block=BasicBlock,
            num_blocks=num_blocks,
            input_dim=input_dim,
            dictionary_dim=dictionary_dim
        )
        
    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device)
        # Pass input through ResNet
        dic = self.resnet(x)
        # Concatenate ones, input, and dictionary output
        y = torch.cat((ones, x), dim=1)
        y = torch.cat((y, dic), dim=1)
        return y
