import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
torch.set_default_dtype(torch.float64)


import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ParamBlockDiagonalKoopman(nn.Module):
    """
    A PyTorch module representing a block diagonal Koopman operator.
    Attributes:
        koopman_dim (int): The dimension of the Koopman operator.
        num_blocks (int): The number of 2x2 blocks in the Koopman operator.
        blocks (nn.ParameterList): A list of parameters representing the angles for each 2x2 block.
        V (nn.Parameter): A parameter representing the matrix V.
    Methods:
        build():
            Initializes the block diagonal Koopman operator with random angles and matrix V.
        forward_K():
            Constructs the block diagonal Koopman operator matrix K using the angles.
        forward_V():
            Returns the matrix V.
    """

    def __init__(self, koopman_dim, v_dim = None):
        super(ParamBlockDiagonalKoopman, self).__init__()
        self.koopman_dim = koopman_dim
        self.num_blocks = self.koopman_dim // 2 
        if v_dim is None:
            self.v_dim = self.koopman_dim
        else:
            self.v_dim = v_dim
        self.build()

    def build(self):
        self.blocks = nn.ParameterList()
        for _ in range(self.num_blocks):
            angle = nn.Parameter(torch.randn(1))
            self.blocks.append(angle)
        if self.koopman_dim % 2 != 0:
            self.blocks.append(nn.Parameter(torch.randn(1)))
        
        self.V = nn.Parameter(torch.randn(self.v_dim, self.koopman_dim))

    def forward_K(self, sample_step = 10):
        device = self.blocks[0].device
        K = torch.zeros(self.koopman_dim, self.koopman_dim, device=device)

        for i in range(self.num_blocks):
            angle = self.blocks[i] * 10 / sample_step
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            K[2 * i, 2 * i] = cos
            K[2 * i, 2 * i + 1] = -sin
            K[2 * i + 1, 2 * i] = sin
            K[2 * i + 1, 2 * i + 1] = cos

        if self.koopman_dim % 2 != 0:
            K[-1, -1] = 1

        return K

    
    def forward_V(self):
        return self.V
    
class TimeEmbeddingBlockDiagonalKoopman(nn.Module):
    """
    A PyTorch module that implements a time embedding block with a block diagonal Koopman operator.
    Attributes:
        koopman_dim (int): The dimension of the Koopman operator.
        inputs (int): The number of input features.
        layers_params (list): A list of parameters for the layers in the dictionary.
        activation (nn.Module): The activation function to use in the dictionary.
    Methods:
        build():
            Builds the trainable dictionary and block diagonal Koopman operator.
        forward(x):
            Performs a forward pass through the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying the Koopman operator.
        dictionary_V(x):
            Computes the dictionary and applies the V matrix of the Koopman operator.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying the V matrix.
    """

    def __init__(self, dictionary_dim, x_dim, u_dim, num_blocks):
        super(TimeEmbeddingBlockDiagonalKoopman, self).__init__()
        self.dictionary_dim = dictionary_dim
        self.koopman_dim = dictionary_dim + 1 + x_dim
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.num_blocks = num_blocks
        self.build()

    def build(self):
        # Initialize the TrainableDictionary with the updated signature
        self.dictionary = TrainableDictionary(
            inputs_dim=self.x_dim,
            dictionary_dim=self.dictionary_dim,
            num_blocks=self.num_blocks
        )
        # Initialize the Koopman operator
        self.koopman = BlockDiagonalKoopman(self.koopman_dim)
    
    def forward(self, x_dic, u_dim, sample_step=10):
        K = self.koopman.forward_K(sample_step)
        V = self.koopman.forward_V()
        y = torch.matmul(x_dic, V)
        y = torch.matmul(y, K)
        # print(f"y.device: {y.device}, K.device: {K.device}")
        return y
        
    
    def dictionary_forward(self, x, sample_step=10):
        x_dic = self.dictionary(x)
        y = self.forward(x_dic, sample_step)
        return y
    
    def dictionary_V(self, x):
        x_dic = self.dictionary(x)
        V = self.koopman.forward_V()
        y = torch.matmul(x_dic, V)
        return y
    
    def regularization_loss(self):
        norm = torch.norm(self.koopman.V, p='fro')
        inv_nomr = torch.norm(torch.linalg.pinv(self.koopman.V), p='fro')
        condition_number = norm * inv_nomr
        return 0.00001 * condition_number
    