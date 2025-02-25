import torch
import torch.nn as nn
import numpy as np
from src.param_periodic_koopman import ParamBlockDiagonalKoopmanWithInputs
from src.data import get_evaluation_dataset
from src.args import parse_arguments, read_config_file
import os
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, params_list, dataset):
    device = torch.device("cpu")

    matrix_V = []
    matrix_B = []
    matrix_V_inv = []
    for idx, params in enumerate(params_list):
        params = torch.tensor(params, dtype=torch.float64).to(device)
        params_scaled = dataset._transform_data(params, dataset.params_mean, dataset.params_std)

        _, V = model.A_matrix(params_scaled[0:1, :])
        V_inv = torch.inverse(V)
        B = model.B_matrix(params_scaled[0:1, :])
        matrix_V.append(V.cpu().detach().numpy())
        matrix_B.append(B.cpu().detach().numpy())
        matrix_V_inv.append(V_inv.cpu().detach().numpy())
    
    matrix_V = np.concatenate(matrix_V, axis=0)
    matrix_B = np.concatenate(matrix_B, axis=0)
    matrix_V_inv = np.concatenate(matrix_V_inv, axis=0)
    return matrix_V, matrix_B, matrix_V_inv

def plot_matrix(matrix, save_dir, name):
    mean_matrix= np.mean(matrix, axis=0)
    std_matrix = np.std(matrix, axis=0)
    plt.figure(figsize=(50, 50))
    ax = sns.heatmap(mean_matrix, cmap='coolwarm', annot=True, fmt=".2e", cbar_kws={'label': 'Value'})
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(name+" Mean Matrix")
    plt.savefig(os.path.join(save_dir, name + '_mean.png'))
    plt.close()

    plt.figure(figsize=(50, 50))
    ax = sns.heatmap(std_matrix, cmap='coolwarm', annot=True, fmt=".2e", cbar_kws={'label': 'Value'})
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(name+" Std Matrix")
    plt.savefig(os.path.join(save_dir, name + '_std.png'))
    plt.close()

    
    
    


def main():
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device("cpu")

    state_dim = config['pca_dim']
    inputs_dim, params_dim = config['inputs_dim'], config['params_dim']
    model = ParamBlockDiagonalKoopmanWithInputs(state_dim, config["dictionary_dim"], inputs_dim, config["u_dictionary_dim"], params_dim, config["dictionary_layers"], config["u_layers"], config["A_layers"], config["B_layers"], config["encoder_type"])
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'model_state_dict.pth')))
    model.to(device)

    data_list_train, params_list_train, inputs_list_train, data_list_test, params_list_test, inputs_list_test, dataset = get_evaluation_dataset(config['data_dir'], config['save_dir'], config['validation_split'])

    matrix_V_train, matrix_B_train, matrix_V_inv_train = evaluate_model(model, params_list_train, dataset)
    matrix_V_test, matrix_B_test, matrix_V_inv_test = evaluate_model(model, params_list_test, dataset)

    np.save(os.path.join(config['save_dir'], 'matrix_V_train.npy'), matrix_V_train)
    np.save(os.path.join(config['save_dir'], 'matrix_B_train.npy'), matrix_B_train)
    np.save(os.path.join(config['save_dir'], 'matrix_V_test.npy'), matrix_V_test)
    np.save(os.path.join(config['save_dir'], 'matrix_B_test.npy'), matrix_B_test)

    plot_matrix(matrix_V_train, config['save_dir'], 'V_train')
    plot_matrix(matrix_B_train, config['save_dir'], 'B_train')
    plot_matrix(matrix_V_test, config['save_dir'], 'V_test')
    plot_matrix(matrix_B_test, config['save_dir'], 'B_test')
    plot_matrix(matrix_V_inv_train, config['save_dir'], 'V_inv_train')
    plot_matrix(matrix_V_inv_test, config['save_dir'], 'V_inv_test')


    return


if __name__ == "__main__":
    main()