import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
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


class Transformer_like_Encoder(nn.Module):
    """
    Implements the state encoder as a trainable dictionary using a feedforward network.
    
    Args:
        input_dim (int): Dimension of input data.
        dictionary_dim (int): Dimension of the dictionary output.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of feedforward layers.
        dropout (float): Dropout probability.

    Methods:
        forward(x): Computes the encoded state dictionary.
    """
    def __init__(self, input_dim, dictionary_dim, hidden_dim=64, hidden_ff = 128, num_layers=6, dropout=0.2):
        super(Transformer_like_Encoder, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            FeedForwardLayerConnection(hidden_dim, FeedForward(hidden_dim, hidden_ff), dropout) 
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, dictionary_dim)

    def forward(self, x):
        y = self.input_layer(x)
        y = F.relu(y)
        for layer in self.layers:
            y = layer(y)
        y = self.output_layer(y)
        return y  # Concatenate ones, input x, and dictionary output

class FeedForwardLayerConnection(nn.Module):
    def __init__(self, size, feed_forward, dropout):
        super(FeedForwardLayerConnection, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x):
        return self.sublayer(x, self.feed_forward)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2): 
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
class TrainableDictionary(nn.Module):
    """
    A neural network module that builds a trainable dictionary for input data.
    
    Args:
        input_dim (int): The dimension of input data.
        dictionary_dim (int): The dimension of the dictionary output.
        num_blocks (list[int], optional): Structure of the ResNet (ignored if using StateEncoder).
        encoder_type (str): Choose between "resnet" and "state_encoder".
    """
    def __init__(self, input_dim, dictionary_dim, num_blocks=[2,2,2], encoder_type="resnet"):
        super(TrainableDictionary, self).__init__()
        self.input_dim = input_dim
        self.dictionary_dim = dictionary_dim
        self.encoder_type = encoder_type

        if encoder_type == "resnet":
            self.encoder = ResNet(
                block=BasicBlock,
                num_blocks=num_blocks,
                input_dim=input_dim,
                dictionary_dim=dictionary_dim
            )
        elif encoder_type == "Transformer_like":
            hidden_dim, hidden_ff, num_layers = num_blocks[0], num_blocks[1], num_blocks[2]
            self.encoder = Transformer_like_Encoder(
                input_dim=input_dim,
                dictionary_dim=dictionary_dim,
                hidden_dim=hidden_dim,
                hidden_ff=hidden_ff,
                num_layers=num_layers
            )
        else:
            raise ValueError("encoder_type must be either 'resnet' or 'Transformer_like'")

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device)
        dic = self.encoder(x)
        return torch.cat([ones, x, dic], dim=1)