import torch
import torch.nn as nn
import numpy as np
from src.data import get_evaluation_dataset
from src.args import parse_arguments, read_config_file
torch.set_default_dtype(torch.float64)
from src.param_perodic_koopman import ParamBlockDiagonalKoopmanWithInputs
import os
import numpy as np
import matplotlib.pyplot as plt


custom_palette = [
    (0/255, 0/255, 0/255),         # 黑色 (0, 0, 0)
    (128/255, 128/255, 128/255),   # 灰色 (128, 128, 128)
    (158/255, 129/255, 186/255),   # 紫灰色 (158, 129, 186)
    (216/255, 205/255, 227/255),   # 浅紫色 (216, 205, 227)
    (76/255, 57/255, 107/255),     # 深紫色 (76, 57, 107)
    (183/255, 176/255, 196/255),   # 灰蓝色 (183, 176, 196)
    (115/255, 62/255, 115/255),    # 紫红色 (115, 62, 115)
    (199/255, 178/255, 199/255)    # 灰紫色 (199, 178, 199)
]

def evaluate_model(model, data_list, params_list, inputs_list, dataset, lidation_split=0.2, sample_step=1):
    # Set the device
    device = torch.device('cpu')

    # Load the evaluation dataset

    data_pred_list = []

   # Evaluate the model
    for data, params, inputs in zip(data_list, params_list, inputs_list):
        data = torch.tensor(data).to(device)
        params = torch.tensor(params).to(device)
        inputs = torch.tensor(inputs).to(device)
        data_initial = data[0].unsqueeze(0)
        data_initial = dataset._pca_transform(data_initial)
        params_scaled = dataset._transform_data(params, dataset.params_mean, dataset.params_std)
        inputs_scaled = dataset._transform_data(inputs, dataset.inputs_mean, dataset.inputs_std)
        data_psi_initail = model(data_initial, inputs[0:1, :], params_scaled[0:1,:], sample_step)
        data_psi_pred = torch.zeros(data.shape[0], data_psi_initail.shape[1])
        data_psi_pred[0] = data_psi_initail
        for i in range(1, data.shape[0]):
            data_psi_pred[i] = model(data_psi_pred[i-1].unsqueeze(0), inputs_scaled[i-1:i :], params_scaled[i-1:i, :], sample_step)

        _, V = model.A_matrix(params_scaled[0:1, :])
        V_inv = torch.inverse(V)
        data_pred = torch.mm(data_psi_pred, V_inv)
        data_pred = dataset._inverse_pca_transform(data_pred)
        data_pred_list.append(data_pred)
    
    return data_list, data_pred_list

def calculate_relative_diff(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, axis=1, ord=2)
    max_norm = np.max(np.linalg.norm(x_true, axis=1, ord=2))
    relative_diff = row_norm_diff / max_norm
    return relative_diff

def calculate_mean_relative_diff_set(x_true_traj, x_pred_traj):
    relative_diffs = [calculate_relative_diff(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)]
    mean_relative_diffs = np.mean(relative_diffs, axis=0)
    return mean_relative_diffs

def calculate_relative_error(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, ord='fro')
    total_norm_true = np.linalg.norm(x_true, ord='fro')
    return row_norm_diff / total_norm_true

def calculate_mean_relative_error_set(x_true_traj, x_pred_traj):
    relative_errors = [calculate_relative_error(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)]
    return relative_errors


def plot_trajectories(x_true_traj, x_pred_traj, labels, filename):
    traj_num = 3
    indices_num = 3
    plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
    })
    random_traj = sorted(np.random.choice(range(len(x_true_traj)), traj_num, replace=False))
    random_indices = sorted(np.random.choice(x_true_traj[0].shape[1], indices_num, replace=False))

    fig, axs = plt.subplots(traj_num, indices_num, figsize=(indices_num * 4, traj_num * 3))

    labels_plot = [
            'True Trajectory',
            'Predicted Trajectory'
        ]

    for i in range(traj_num):
        traj = random_traj[i]
        idx = random_indices[0]

        y_all = np.concatenate([
            x_true_traj[traj][:, idx],
                x_pred_traj[traj][:, idx],
            ])
        y_min, y_max = np.min(y_all), np.max(y_all)

        axs[i, 0].plot(x_true_traj[traj][:, idx], label=labels_plot[0], color=custom_palette[0])
        axs[i, 0].plot(x_pred_traj[traj][:, idx], label=labels_plot[1], color=custom_palette[1])
        axs[i, 0].set_title(f'Traj {traj}, All Models')
        axs[i, 0].set_xlabel('Prediction Step')
        axs[i, 0].set_ylabel('$\Phi_{loop}$')
        axs[i, 0].set_ylim(y_min, y_max)

        axs[i, 1].plot(x_true_traj[traj][:, idx], label=labels_plot[0], color=custom_palette[0])
        axs[i, 1].set_title(f'Traj {traj}, True Trajectory')
        axs[i, 1].set_xlabel('Prediction Step')
        axs[i, 1].set_ylim(y_min, y_max)
        axs[i, 1].tick_params(labelleft=False)
        axs[i, 1].set_ylabel('')

        axs[i, 2].plot(x_pred_traj[traj][:, idx], label=labels_plot[4], color=custom_palette[4])
        axs[i, 2].set_title(f'Traj {traj}, Predicted Trajectory')
        axs[i, 2].set_xlabel('Prediction Step')
        axs[i, 2].set_ylim(y_min, y_max)
        axs[i, 2].tick_params(labelleft=False)
        axs[i, 2].set_ylabel('')

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), title="Legend", fontsize='large', title_fontsize='x-large')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename)
    
def main():
    # Load configuration
    args = parse_arguments()
    config = read_config_file(args.config)

    # Loss history
    loss_history = np.load(os.path.join(config["save_dir"], "losses.pth"))
    plt.figure()
    plt.plot(loss_history['train_losses'], label='Train', color=custom_palette[2])
    plt.plot(loss_history['test_losses'], label='Test', color=custom_palette[3])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(os.path.join(config["save_dir"], "losses.png"))

    # Load the model
    model = torch.load(os.path.join(config["save_dir"], "model.pth"))

    # Evaluate the model

    data_list_train, params_list_train, inputs_list_train, data_list_test, params_list_test, inputs_list_test, dataset = get_evaluation_dataset(config['data_dir'], config['save_dir'], config['validation_split'])

    data_list_train, data_pred_list_train = evaluate_model(model, data_list_train, params_list_train, inputs_list_train, dataset, sample_step=1)

    data_list_test, data_pred_list_test = evaluate_model(model, data_list_test, params_list_test, inputs_list_test, dataset, sample_step=1)

    # Calculate the mean relative difference
    mean_relative_diffs_train = calculate_mean_relative_diff_set(data_list_train, data_pred_list_train)
    mean_relative_diffs_test = calculate_mean_relative_diff_set(data_list_test, data_pred_list_test)

    plt.figure()
    plt.plot(mean_relative_diffs_train, label='Train', color=custom_palette[2])
    plt.plot(mean_relative_diffs_test, label='Test', color=custom_palette[3])
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Mean Relative Difference')
    plt.yscale('log')
    plt.savefig(os.path.join(config['save_dir'], 'mean_relative_diff.png'))

    # Calculate the mean relative error
    mean_relative_errors_train = calculate_mean_relative_error_set(data_list_train, data_pred_list_train)
    mean_relative_errors_test = calculate_mean_relative_error_set(data_list_test, data_pred_list_test)
    # Plot the mean relative error
    plt.figure()
    plt.boxplot([mean_relative_errors_train, mean_relative_errors_test], labels=['Train', 'Test'], color=[custom_palette[2], custom_palette[3]])
    plt.ylabel('Mean Relative Error')
    plt.yscale('log')
    plt.savefig(os.path.join(config['save_dir'], 'mean_relative_error.png'))

    # Plot the trajectories
    plot_trajectories(data_list_train, data_pred_list_train, ['Train'], os.path.join(config['save_dir'], 'trajectories.png'))

    # Save the results
    results = {
        'mean_relative_diffs_train': mean_relative_diffs_train,
        'mean_relative_diffs_test': mean_relative_diffs_test,
        'mean_relative_errors_train': mean_relative_errors_train,
        'mean_relative_errors_test': mean_relative_errors_test
    }
    np.save(os.path.join(config['save_dir'], 'results.npy'), results)

    results = {
        'data_list_test': data_list_test,
        'data_pred_list_test': data_pred_list_test
    }
    np.save(os.path.join(config['save_dir'], 'data.npy'), results)

    return

if __name__ == '__main__':
    main()


    

        
        

