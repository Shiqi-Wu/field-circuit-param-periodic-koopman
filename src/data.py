import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.decomposition import PCA

import os
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        return

    def _build_training_dataset(self, data_list, params_list, inputs_list, step_size, pca_dim):
        data_list = [torch.tensor(d, dtype=torch.float64) for d in data_list]
        params_list = [torch.tensor(p, dtype=torch.float64) for p in params_list]
        inputs_list = [torch.tensor(i, dtype=torch.float64) for i in inputs_list]

        self.params_mean, self.params_std, params_list = self._fit_transform_data_list(params_list)
        self.inputs_mean, self.inputs_std, inputs_list = self._fit_transform_data_list(inputs_list)
        self.data_mean, self.data_std, self.data_pca_mean, self.data_pca_std, data_pca_list, self.pca = self._pca_fit_transform(data_list, pca_dim)

        # print(params_list[0])

        data_slices, params_slices, inputs_slices = self._cut_slices(data_pca_list, params_list, inputs_list, step_size)
        return data_slices, params_slices, inputs_slices

    def _build_testing_dataset(self, data_list, params_list, inputs_list, step_size):
        data_list = [torch.tensor(d, dtype=torch.float64) for d in data_list]
        params_list = [torch.tensor(p, dtype=torch.float64) for p in params_list]
        inputs_list = [torch.tensor(i, dtype=torch.float64) for i in inputs_list]

        params_list = self._transform_data_list(params_list, self.params_mean, self.params_std)
        inputs_list = self._transform_data_list(inputs_list, self.inputs_mean, self.inputs_std)
        data_pca_list = self._pca_transform_list(data_list)
        # print(params_list[0])

        data_slices, params_slices, inputs_slices = self._cut_slices(data_pca_list, params_list, inputs_list, step_size)
        return data_slices, params_slices, inputs_slices

    def _pca_fit_transform(self, data_list, pca_dim):
        data = torch.cat(data_list, dim=0)
        # print(data.shape)
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data = (data - data_mean) / data_std
        # print(data_mean)
        # print(data_std)

        pca = PCA(n_components=pca_dim)
        data_np = data.numpy()
        data_pca_np = pca.fit_transform(data_np.reshape(-1, data_np.shape[-1]))
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = np.sum(explained_variance_ratio)
        print(f"Explained variance by PCA: {explained_variance * 100:.2f}%")

        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(explained_variance_ratio), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()
        data_pca = torch.tensor(data_pca_np, dtype=torch.float64)
        data_pca_mean = data_pca.mean(dim=0)
        data_pca_std = data_pca.std(dim=0)

        data_pca_list = []
        start_idx = 0
        for d in data_list:
            end_idx = start_idx + d.size(0)
            data_pca_list.append(data_pca[start_idx:end_idx])
            start_idx = end_idx

        data_pca_list = [(d - data_pca_mean) / data_pca_std for d in data_pca_list]

        return data_mean, data_std, data_pca_mean, data_pca_std, data_pca_list, pca

    def _pca_transform(self, data):
        data = (data - self.data_mean) / self.data_std
        data_pca_np = self.pca.transform(data.numpy())
        data_pca = torch.tensor(data_pca_np, dtype=torch.float32)
        return (data_pca - self.data_pca_mean) / self.data_pca_std

    def _pca_transform_list(self, data_list):
        return [self._pca_transform(d) for d in data_list]

    def _inverse_pca_transform(self, data_pca):
        data_pca = data_pca * self.data_pca_std + self.data_pca_mean
        data_np = self.pca.inverse_transform(data_pca.numpy())
        data = torch.tensor(data_np, dtype=torch.float32)
        return data * self.data_std + self.data_mean

    def _inverse_pca_transform_list(self, data_pca_list):
        return [self._inverse_pca_transform(d) for d in data_pca_list]

    def _fit_transform_data_list(self, data_list, epsilon=1e-8):
        data = torch.cat([d.unsqueeze(0) for d in data_list], dim=0)
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data_std[data_std < epsilon] = 1
        data_list = [(d - data_mean) / data_std for d in data_list]
        return data_mean, data_std, data_list

    def _transform_data(self, data, data_mean, data_std, epsilon=1e-8):
        # data_std[data_std < epsilon] = 1
        return (data - data_mean) / data_std

    def _transform_data_list(self, data_list, data_mean, data_std):
        return [self._transform_data(d, data_mean, data_std) for d in data_list]

    def _inverse_transform_data(self, data, data_mean, data_std):
        return data * data_std + data_mean

    def _inverse_transform_data_list(self, data_list, data_mean, data_std):
        return [d * data_std + data_mean for d in data_list]

    def _cut_slices(self, data_list, params_list, inputs_list, step_size):
        data_slices, params_slices, inputs_slices = [], [], []
        for data, params, inputs in zip(data_list, params_list, inputs_list):
            for i in range(data.size(0) - step_size):
                data_slices.append(data[i:i+step_size].unsqueeze(0))
                params_slices.append(params[i:i+step_size].unsqueeze(0))
                inputs_slices.append(inputs[i:i+step_size].unsqueeze(0))
        data_slices = torch.cat(data_slices, dim=0)
        params_slices = torch.cat(params_slices, dim=0)
        inputs_slices = torch.cat(inputs_slices, dim=0)
        
        return data_slices, params_slices, inputs_slices


def get_dataset(data_dir, step_size, pca_dim, batch_size=256, validation_split=0.2):
    data_list, params_list, inputs_list = [], [], []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            ff = np.load(os.path.join(data_dir, file), allow_pickle=True)
            ff = ff.item()

            data_list.append(ff['data'])
            params_list.append(ff['params'])
            inputs_list.append(ff['inputs'])

    data_list_train, data_list_test, params_list_train, params_list_test, inputs_list_train, inputs_list_test = train_test_split(
        data_list, params_list, inputs_list, test_size=validation_split, random_state=42)

    dataset = CustomDataset()
    # training_data, training_params, training_inputs = dataset._build_training_dataset(data_list_train, params_list_train, inputs_list_train, step_size, pca_dim)
    training_data, training_params, training_inputs = dataset._build_training_dataset(data_list_train, params_list_train, inputs_list_train, step_size, pca_dim)
    testing_data, testing_params, testing_inputs = dataset._build_testing_dataset(data_list_test, params_list_test, inputs_list_test, step_size)
    # print(training_inputs[0])
    # print(testing_inputs[0])

    training_data = torch.tensor(training_data, dtype=torch.float64)
    training_params = torch.tensor(training_params, dtype=torch.float64)
    training_inputs = torch.tensor(training_inputs, dtype=torch.float64)
    testing_data = torch.tensor(testing_data, dtype=torch.float64)
    testing_params = torch.tensor(testing_params, dtype=torch.float64)
    testing_inputs = torch.tensor(testing_inputs, dtype=torch.float64)
    indices = torch.randperm(training_data.size(0))
    training_data = training_data[indices]
    training_params = training_params[indices]
    training_inputs = training_inputs[indices]
    training_dataset = torch.utils.data.TensorDataset(training_data, training_params, training_inputs)
    testing_dataset = torch.utils.data.TensorDataset(testing_data, testing_params, testing_inputs)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    return training_loader, testing_loader, dataset

def get_evaluation_dataset(data_dir, save_dir, validation_split=0.2):
    data_list, params_list, inputs_list = [], [], []
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            ff = np.load(os.path.join(data_dir, file), allow_pickle=True)
            ff = ff.item()
            
            data_list.append(ff['data'])
            params_list.append(ff['params'])
            inputs_list.append(ff['inputs'])

    data_list_train, data_list_test, params_list_train, params_list_test, inputs_list_train, inputs_list_test = train_test_split(
        data_list, params_list, inputs_list, test_size=validation_split, random_state=42)
    
    dataset_dir = os.path.join(save_dir, "dataset.pth")
    dataset = torch.load(dataset_dir, weights_only=False)

    return data_list_train, params_list_train, inputs_list_train, data_list_test, params_list_test, inputs_list_test, dataset    
