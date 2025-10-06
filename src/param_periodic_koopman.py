import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
torch.set_default_dtype(torch.float64)


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resnet import TrainableDictionary, BasicBlock, ResNet, BasicBlock2

class ParamMatrix(nn.Module):
    def __init__(self, input_dim, matric_dim_1,  matric_dim_2, layers, BN = True):
        super(ParamMatrix, self).__init__()
        self.input_dim = input_dim
        self.matric_dim_1 = matric_dim_1
        self.matric_dim_2 = matric_dim_2
        self.layers = layers
        self.BN = BN
        self.build()
    
    def build(self):
        output_dim = self.matric_dim_1 * self.matric_dim_2
        if self.BN == True:
            self.resnet = ResNet(BasicBlock, self.layers, self.input_dim, output_dim)
        else:
            self.resnet = ResNet(BasicBlock2, self.layers, self.input_dim, output_dim)
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, self.matric_dim_1, self.matric_dim_2)
        return x
    
    def initialize_to_zero(self):
        for m in self.resnet.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()
    
class ParamBlockDiagonalMatrix(nn.Module):

    def __init__(self, input_dim, matric_dim, layers, BN = True):
        super(ParamBlockDiagonalMatrix, self).__init__()
        self.input_dim = input_dim
        self.matric_dim = matric_dim
        self.layers = layers
        self.num_blocks = matric_dim // 2
        self.BN = BN
        self.build()

    def build(self):
        self.output_dim = self.matric_dim * self.matric_dim + self.num_blocks
        if self.BN == True:
            self.resnet = ResNet(BasicBlock, self.layers, self.input_dim, self.output_dim)
        else:
            self.resnet = ResNet(BasicBlock2, self.layers, self.input_dim, self.output_dim)

    def forward_K(self, resnet_output, sample_step=1):
        # Get the device (e.g., GPU or CPU)
        device = resnet_output.device
        
        # Slice the resnet output to get the K parameters of shape [batch_size, num_blocks]
        K_params = resnet_output[:, self.matric_dim * self.matric_dim:]  # Shape: [batch_size, num_blocks]
    
        # Compute the angles, cos, and sin values for each block in the batch
        angles = K_params * sample_step  # Shape: [batch_size, num_blocks]
        cos_vals = torch.cos(angles)  # Shape: [batch_size, num_blocks]
        sin_vals = torch.sin(angles)  # Shape: [batch_size, num_blocks]

        # Initialize the K tensor with zeros of shape [batch_size, koopman_dim, koopman_dim]
        batch_size = K_params.shape[0]
        K = torch.zeros(batch_size, self.matric_dim, self.matric_dim, device=device)

        # Create indices for the blocks to fill the rotation matrices
        indices = torch.arange(self.num_blocks, device=device)
        i_indices = 2 * indices  # Row indices for the first element of each block
        j_indices = i_indices + 1  # Row indices for the second element of each block

        # Fill the diagonal blocks of the K matrices
        K[:, i_indices, i_indices] = cos_vals  # K[batch_size, 2i, 2i] = cos
        K[:, i_indices, j_indices] = -sin_vals  # K[batch_size, 2i, 2i+1] = -sin
        K[:, j_indices, i_indices] = sin_vals  # K[batch_size, 2i+1, 2i] = sin
        K[:, j_indices, j_indices] = cos_vals  # K[batch_size, 2i+1, 2i+1] = cos

        # Handle the case where self.matric_dim is odd
        # Set the last diagonal element of each K matrix to 1
        if self.matric_dim % 2 != 0:
            K[:, -1, -1] = 1

        return K  # Output shape: [batch_size, self.matric_dim, self.matric_dim]

 
    def forward_V(self, resnet_output):
        return resnet_output[:, :self.matric_dim * self.matric_dim].view(-1, self.matric_dim, self.matric_dim)

    def forward(self, input, sample_step=1):
        resnet_output = self.resnet(input)
        K = self.forward_K(resnet_output, sample_step)
        V = self.forward_V(resnet_output)
        return K, V

class ParamBlockDiagonalMatrixWoV(nn.Module):

    def __init__(self, input_dim, matric_dim, layers, BN = True):
        super(ParamBlockDiagonalMatrixWoV, self).__init__()
        self.input_dim = input_dim
        self.matric_dim = matric_dim
        self.layers = layers
        self.num_blocks = matric_dim // 2
        self.BN = BN
        self.build()

    def build(self):
        if self.BN == True:
            self.resnet = ResNet(BasicBlock, self.layers, self.input_dim, self.num_blocks)
        else:
            self.resnet = ResNet(BasicBlock2, self.layers, self.input_dim, self.num_blocks)

    def forward(self, params, sample_step=1):
        # Get the device (e.g., GPU or CPU)
        resnet_output = self.resnet(params)
        device = resnet_output.device

        K_params = resnet_output  # Shape: [batch_size, num_blocks]
        batch_size, num_blocks = K_params.shape
        assert num_blocks == self.num_blocks, \
            f"K_params must have num_blocks={self.num_blocks}, got {num_blocks}"

        # Compute the angles, cos, and sin values for each block in the batch
        angles = K_params * sample_step  # Shape: [batch_size, num_blocks]
        cos_vals = torch.cos(angles)  # Shape: [batch_size, num_blocks]
        sin_vals = torch.sin(angles)  # Shape: [batch_size, num_blocks]

        # Initialize the K tensor with zeros of shape [batch_size, matric_dim, matric_dim]
        assert self.matric_dim >= 2 * self.num_blocks, \
            "matric_dim must be at least 2 * num_blocks to accommodate rotation matrices"
        K = torch.zeros(batch_size, self.matric_dim, self.matric_dim, device=device)

        # Create indices for the blocks to fill the rotation matrices
        indices = torch.arange(self.num_blocks, device=device)
        i_indices = 2 * indices  # Row indices for the first element of each block
        j_indices = i_indices + 1  # Row indices for the second element of each block

        # Fill the diagonal blocks of the K matrices
        K[:, i_indices, i_indices] = cos_vals  # K[batch_size, 2i, 2i] = cos
        K[:, i_indices, j_indices] = -sin_vals  # K[batch_size, 2i, 2i+1] = -sin
        K[:, j_indices, i_indices] = sin_vals  # K[batch_size, 2i+1, 2i] = sin
        K[:, j_indices, j_indices] = cos_vals  # K[batch_size, 2i+1, 2i+1] = cos

        return K
    
class ParamSkewSymmetricMatrix(nn.Module):
    def __init__(self, params_dim, koopman_dim, layers, BN = True):
        super(ParamSkewSymmetricMatrix, self).__init__()
        self.params_dim = params_dim
        self.koopman_dim = koopman_dim
        self.layers = layers
        self.BN = BN
        self.build()

    def build(self):
        if self.BN == True:
            self.resnet = ResNet(BasicBlock, self.layers, self.params_dim, self.koopman_dim * (self.koopman_dim - 1) // 2)
        else:
            self.resnet = ResNet(BasicBlock2, self.layers, self.params_dim, self.koopman_dim * (self.koopman_dim - 1) // 2)

    def forward(self, params):
        skew_symmetric_params = self.resnet(params)
        batch_size = skew_symmetric_params.shape[0]
        skew_symmetric_matrix = torch.zeros(batch_size, self.koopman_dim, self.koopman_dim, device=skew_symmetric_params.device)
        indices = torch.tril_indices(self.koopman_dim, self.koopman_dim, offset=-1)
        skew_symmetric_matrix[:, indices[0], indices[1]] = skew_symmetric_params
        skew_symmetric_matrix = skew_symmetric_matrix - skew_symmetric_matrix.transpose(1, 2)
        return skew_symmetric_matrix
    

class ParamSpecialOrthogonalMatrix(nn.Module):
    def __init__(self, params_dim, koopman_dim, layers, BN = True):
        super(ParamSpecialOrthogonalMatrix, self).__init__()
        self.params_dim = params_dim
        self.koopman_dim = koopman_dim
        self.layers = layers
        self.BN = BN
        self.build()

    def build(self):
        self.skew_symmetric_matrix = ParamSkewSymmetricMatrix(self.params_dim, self.koopman_dim, self.layers, BN = self.BN)
    
    def forward(self, params):
        skew_symmetric_matrix = self.skew_symmetric_matrix(params)
        special_orthogonal_matrix = torch.matrix_exp(skew_symmetric_matrix)
        return special_orthogonal_matrix
    
class ParamBlockTriangularMatrix(nn.Module):
    def __init__(self, params_dim, koopman_dim, layers, BN = True):
        super(ParamBlockTriangularMatrix, self).__init__()
        self.params_dim = params_dim
        self.koopman_dim = koopman_dim
        self.layers = layers
        self.BN = BN
        self.build()

    def build(self):
        if self.BN == True:
            self.resnet = ResNet(BasicBlock, self.layers, self.params_dim, self.koopman_dim * (self.koopman_dim - 1) // 2)
        else:
            self.resnet = ResNet(BasicBlock2, self.layers, self.params_dim, self.koopman_dim * (self.koopman_dim - 1) // 2)
    
    def forward(self, params):
        block_triangular_params = self.resnet(params)
        diag_params = block_triangular_params[:, :self.koopman_dim // 2]
        off_diag_params = block_triangular_params[:, self.koopman_dim // 2:]
    
        batch_size = block_triangular_params.shape[0]
    
        # 初始化 K 张量，形状为 [batch_size, koopman_dim, koopman_dim]
        K = torch.zeros(batch_size, self.koopman_dim, self.koopman_dim, device=block_triangular_params.device)
    
        # 填充对角块：每个 2×2 块构成一个旋转矩阵
        for i in range(self.koopman_dim // 2):
            cos_val = torch.cos(diag_params[:, i])
            sin_val = torch.sin(diag_params[:, i])
            K[:, 2 * i, 2 * i] = cos_val
            K[:, 2 * i, 2 * i + 1] = -sin_val
            K[:, 2 * i + 1, 2 * i] = sin_val
            K[:, 2 * i + 1, 2 * i + 1] = cos_val

        # 构造上三角区域的掩码（不包含主对角线）
        mask = torch.triu(torch.ones(self.koopman_dim, self.koopman_dim, dtype=torch.bool, device=K.device), diagonal=1)
        # 去除各个 2×2 块内部的上三角位置，即 (2*i, 2*i+1) 对应的位置
        for i in range(self.koopman_dim // 2):
            mask[2 * i, 2 * i + 1] = False

        # 获取满足条件的位置索引：既在上三角中，又不属于对角块内
        non_block_upper_indices = torch.nonzero(mask, as_tuple=True)
    
        # 将 off_diag_params 填充到 K 中这些位置
        # 注意：off_diag_params 的元素个数必须与 mask 中 True 的数量一致
        K[:, non_block_upper_indices[0], non_block_upper_indices[1]] = off_diag_params

        return K


class ParamBlockDiagonalKoopmanWithInputs(nn.Module):

    def __init__(self, state_dim, dictionary_dim, inputs_dim, u_dictionary_dim, params_dim, dictionary_layers, u_layers, A_layers, B_layers, dictionary_type = "resnet"):
        super(ParamBlockDiagonalKoopmanWithInputs, self).__init__()
        self.state_dim = state_dim
        self.dictionary_dim = dictionary_dim
        self.inputs_dim = inputs_dim
        self.u_dictionary_dim = u_dictionary_dim
        self.params_dim = params_dim
        self.dictionary_layers = dictionary_layers
        self.u_layers = u_layers
        self.A_layers = A_layers
        self.B_layers = B_layers
        self.dictionary_type = dictionary_type
        self.build()

    def build(self):
        self.dictionary = TrainableDictionary(self.state_dim, self.dictionary_dim, self.dictionary_layers, encoder_type=self.dictionary_type)
        self.u_dictionary = TrainableDictionary(self.inputs_dim, self.u_dictionary_dim, self.u_layers, encoder_type=self.dictionary_type)
        self.koopman_dim = self.dictionary_dim + self.state_dim + 1
        self.A_matrix = ParamBlockDiagonalMatrix(self.params_dim, self.koopman_dim, self.A_layers, BN = False)
        self.u_dim = self.u_dictionary_dim + self.inputs_dim + 1
        self.B_matrix = ParamMatrix(self.params_dim, self.u_dim,self.koopman_dim, self.B_layers, BN = False)

    def forward(self, x_dic, inputs_dic, params, sample_step=1):
        K, _ = self.A_matrix(params, sample_step)
        B = self.B_matrix(params)

        # Expand dimensions of x_dic to match K and V for batch matrix multiplication
        x_dic = x_dic.unsqueeze(1)  # Shape: (batch_size, 1, koopman_dim)
        x_dic = torch.matmul(x_dic, K)

        x_dic = x_dic.squeeze(1)  # Shape: (batch_size, koopman_dim)

        inputs_dic = inputs_dic.unsqueeze(1)  # Shape: (batch_size, 1, inputs_dim)
        inputs_dic = torch.matmul(inputs_dic, B)
        inputs_dic = inputs_dic.squeeze(1)

        results = x_dic + inputs_dic

        return results
    
    def dictionary_V(self, x, params, sample_step=1):
        x_dic = self.dictionary(x)
        _, V = self.A_matrix(params, sample_step)
        x_dic = x_dic.unsqueeze(1)
        x_dic = torch.matmul(x_dic, V)
        x_dic = x_dic.squeeze(1)
        return x_dic
    
    # def regularization_loss(self):
    #     norm = torch.norm(self.koopman.V, p='fro')
    #     inv_nomr = torch.norm(torch.linalg.pinv(self.koopman.V), p='fro')
    #     condition_number = norm * inv_nomr
    #     return 0.00001 * condition_number

    def regularization_loss(self, params, sample_step=1):
        _, V = self.A_matrix(params, sample_step)
    
        # 1-范数
        norm_V = torch.norm(V, p=1, dim=(1, 2))
    
        # 解线性方程组近似逆范数
        b = torch.ones(V.shape[0], V.shape[1], 1, device=V.device)  # 假设列数相同
        V_inv_approx = torch.linalg.solve(V, b)
        norm_V_inv = torch.norm(V_inv_approx, p=2, dim=(1, 2))
    
        # 条件数近似
        cond_number = norm_V * norm_V_inv

        reg_loss = torch.log1p(cond_number).mean() - torch.log(torch.tensor(10.0, device=V.device))


        # ReLU 和 log 约束（避免极端值）
        return reg_loss.clamp(min=0.0), norm_V_inv



class ParamBlockDiagonalKoopmanWithInputs3NetWorks(nn.Module):

    def __init__(self, state_dim, dictionary_dim, inputs_dim, u_dictionary_dim, params_dim, dictionary_layers, u_layers, Lambda_layers, V_layers, B_layers, dictionary_type = "resnet"):
        super(ParamBlockDiagonalKoopmanWithInputs3NetWorks, self).__init__()
        self.state_dim = state_dim
        self.dictionary_dim = dictionary_dim
        self.inputs_dim = inputs_dim
        self.u_dictionary_dim = u_dictionary_dim
        self.params_dim = params_dim
        self.dictionary_layers = dictionary_layers
        self.u_layers = u_layers
        self.Lambda_layers = Lambda_layers
        self.V_layers = V_layers
        self.B_layers = B_layers
        self.dictionary_type = dictionary_type
        self.build()

    def build(self):
        self.dictionary = TrainableDictionary(self.state_dim, self.dictionary_dim, self.dictionary_layers, encoder_type=self.dictionary_type)
        self.u_dictionary = TrainableDictionary(self.inputs_dim, self.u_dictionary_dim, self.u_layers, encoder_type=self.dictionary_type)
        self.koopman_dim = self.dictionary_dim + self.state_dim + 1
        self.Lambda_matrix = ParamBlockDiagonalMatrixWoV(self.params_dim, self.koopman_dim, self.Lambda_layers, BN = False)
        self.V_matrix = ParamMatrix(self.params_dim, self.koopman_dim, self.koopman_dim, self.V_layers, BN = False)
        self.u_dim = self.u_dictionary_dim + self.inputs_dim + 1
        self.B_matrix = ParamMatrix(self.params_dim, self.u_dim, self.koopman_dim, self.B_layers, BN = False)

    def forward(self, x_dic, inputs_dic, params, sample_step=1):
        Lambda = self.Lambda_matrix(params, sample_step)
        B = self.B_matrix(params)

        # Expand dimensions of x_dic to match K and V for batch matrix multiplication
        x_dic = x_dic.unsqueeze(1)  # Shape: (batch_size, 1, koopman_dim)
        x_dic = torch.matmul(x_dic, Lambda)

        x_dic = x_dic.squeeze(1)  # Shape: (batch_size, koopman_dim)

        inputs_dic = inputs_dic.unsqueeze(1)  # Shape: (batch_size, 1, inputs_dim)
        inputs_dic = torch.matmul(inputs_dic, B)
        inputs_dic = inputs_dic.squeeze(1)

        results = x_dic + inputs_dic

        return results
    
    def dictionary_V(self, x, params):
        x_dic = self.dictionary(x)
        V = self.V_matrix(params)
        x_dic = x_dic.unsqueeze(1)
        x_dic = torch.matmul(x_dic, V)
        x_dic = x_dic.squeeze(1)
        return x_dic
    
    # def regularization_loss(self):
    #     norm = torch.norm(self.koopman.V, p='fro')
    #     inv_nomr = torch.norm(torch.linalg.pinv(self.koopman.V), p='fro')
    #     condition_number = norm * inv_nomr
    #     return 0.00001 * condition_number

    def regularization_loss(self, params):
        V = self.V_matrix(params)
    
        # 1-范数
        norm_V = torch.norm(V, p=1, dim=(1, 2))
    
        # 解线性方程组近似逆范数
        b = torch.ones(V.shape[0], V.shape[1], 1, device=V.device)  # 假设列数相同
        norm_V_inv = torch.linalg.solve(V, b).abs().sum(dim=(1, 2))  # 近似 1-范数的逆范数
    
        # 条件数近似
        cond_number = norm_V * norm_V_inv

        # ReLU 和 log 约束（避免极端值）
        return 1 * F.relu(torch.log(cond_number).mean() - torch.log(torch.tensor(10.0, device=V.device))), norm_V_inv

class ParamKoopmanWithInputs(nn.Module):

    def __init__(self, state_dim, dictionary_dim, inputs_dim, u_dictionary_dim, params_dim, dictionary_layers, u_layers, A_layers, B_layers, dictionary_type = "resnet"):
        super(ParamKoopmanWithInputs, self).__init__()
        self.state_dim = state_dim
        self.dictionary_dim = dictionary_dim
        self.inputs_dim = inputs_dim
        self.u_dictionary_dim = u_dictionary_dim
        self.params_dim = params_dim
        self.dictionary_layers = dictionary_layers
        self.u_layers = u_layers
        self.A_layers = A_layers
        self.B_layers = B_layers
        self.dictionary_type = dictionary_type
        self.build()

    def build(self):
        self.dictionary = TrainableDictionary(self.state_dim, self.dictionary_dim, self.dictionary_layers, encoder_type=self.dictionary_type)
        self.u_dictionary = TrainableDictionary(self.inputs_dim, self.u_dictionary_dim, self.u_layers, encoder_type=self.dictionary_type)
        self.koopman_dim = self.dictionary_dim + self.state_dim + 1
        self.A_matrix = ParamMatrix(self.params_dim, self.koopman_dim, self.koopman_dim, self.A_layers, BN = False)
        self.u_dim = self.u_dictionary_dim + self.inputs_dim + 1
        self.B_matrix = ParamMatrix(self.params_dim, self.u_dim,self.koopman_dim, self.B_layers, BN = False)

    def forward(self, x_dic, inputs_dic, params):
        A = self.A_matrix(params)
        B = self.B_matrix(params)

        # Expand dimensions of x_dic to match K and V for batch matrix multiplication
        x_dic = x_dic.unsqueeze(1)  # Shape: (batch_size, 1, koopman_dim)
        x_dic = torch.matmul(x_dic, A)

        x_dic = x_dic.squeeze(1)  # Shape: (batch_size, koopman_dim)

        inputs_dic = inputs_dic.unsqueeze(1)  # Shape: (batch_size, 1, inputs_dim)
        inputs_dic = torch.matmul(inputs_dic, B)
        inputs_dic = inputs_dic.squeeze(1)

        results = x_dic + inputs_dic

        return results
    
    def coordinate_projection(self, x_dic):
        x_dic = x_dic[:, 1:self.state_dim+1]
        return x_dic
    
class ParamOrthogonalKoopmanWithInputs(nn.Module):
    def __init__(self, state_dim, dictionary_dim, inputs_dim, u_dictionary_dim, params_dim, dictionary_layers, u_layers, Q_layers, T_layers, B_layers, dictionary_type = "resnet"):
        super(ParamOrthogonalKoopmanWithInputs, self).__init__()
        self.state_dim = state_dim
        self.dictionary_dim = dictionary_dim
        self.inputs_dim = inputs_dim
        self.u_dictionary_dim = u_dictionary_dim
        self.params_dim = params_dim
        self.dictionary_layers = dictionary_layers
        self.u_layers = u_layers
        self.Q_layers = Q_layers
        self.T_layers = T_layers
        self.B_layers = B_layers
        self.dictionary_type = dictionary_type
        self.build()

    def build(self):
        self.dictionary = TrainableDictionary(self.state_dim, self.dictionary_dim, self.dictionary_layers, encoder_type=self.dictionary_type)
        self.u_dictionary = TrainableDictionary(self.inputs_dim, self.u_dictionary_dim, self.u_layers, encoder_type=self.dictionary_type)
        self.koopman_dim = self.dictionary_dim + self.state_dim + 1
        self.u_dim = self.u_dictionary_dim + self.inputs_dim + 1
        self.Q_matrix = ParamSpecialOrthogonalMatrix(self.params_dim, self.koopman_dim, self.Q_layers, BN = False)
        self.Lambda_matrix = ParamBlockDiagonalMatrixWoV(self.params_dim, self.koopman_dim, self.T_layers, BN = False)
        self.B_matrix = ParamMatrix(self.params_dim, self.u_dim,self.koopman_dim, self.B_layers, BN = False)

    def forward(self, x_dic, inputs_dic, params, sample_step = 1):
        K = self.Lambda_matrix(params, sample_step)
        B = self.B_matrix(params)

        # Expand dimensions of x_dic to match K and V for batch matrix multiplication
        x_dic = x_dic.unsqueeze(1)
        x_dic = torch.matmul(x_dic, K)
        x_dic = x_dic.squeeze(1)

        inputs_dic = inputs_dic.unsqueeze(1)  # Shape: (batch_size, 1, inputs_dim)
        inputs_dic = torch.matmul(inputs_dic, B)
        inputs_dic = inputs_dic.squeeze(1)

        results = x_dic + inputs_dic

        results = results.squeeze(1)
        return results
    
    def dictionary_V(self, x, params):
        x_dic = self.dictionary(x)
        V = self.Q_matrix(params)
        x_dic = x_dic.unsqueeze(1)
        x_dic = torch.matmul(x_dic, V)
        x_dic = x_dic.squeeze(1)
        return x_dic