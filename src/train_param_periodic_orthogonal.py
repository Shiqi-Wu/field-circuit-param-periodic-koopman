import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from src.args import parse_arguments, read_config_file
from src.data import get_dataset
from src.param_periodic_koopman import ParamOrthogonalKoopmanWithInputs

def check_for_nan(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
    return False


def koopman_loss(model, x_true, params, inputs):
    # 计算 x_dic_true
    x_dic_true = []
    for i in range(x_true.shape[1]):
        x_dic_true.append(model.dictionary(x_true[:, i, :]))
    x_dic_true = torch.stack(x_dic_true, dim=1)  # [batch_size, sequence_length, feature_dim]
    
    
    L = x_true.shape[1]
    
    # 初始化 y_dic_pred，确保计算图不被切断
    y_dic_pred = [x_dic_true[:, 0, :]]  # 用列表保存序列，避免 inplace 操作
    
    # 逐步预测后续值
    for l in range(L - 1):
        next_pred = model(y_dic_pred[-1], inputs[:, l, :], params[:, l, :])
        y_dic_pred.append(next_pred)  # 添加到列表中
    
    # 将预测值拼接成张量
    y_dic_pred = torch.stack(y_dic_pred, dim=1)  # [batch_size, sequence_length, feature_dim]
    
    mse_loss = F.mse_loss(y_dic_pred, x_dic_true)
    return mse_loss


def train_one_epoch(model, optimizer, train_loader, device, epoch, log_file):
    model.train()
    total_loss = 0.0

    for batch_idx, (x_true, params, inputs) in enumerate(train_loader):
        x_true, params, inputs = x_true.to(device), params.to(device), inputs.to(device)
        optimizer.zero_grad()
        loss = koopman_loss(model, x_true, params, inputs)
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for NaN/Inf values before optimizer step
        if check_for_nan(model):
            with open(log_file, "a") as f:
                f.write(f"Epoch {epoch}, Batch {batch_idx}: NaN/Inf values detected before optimizer step\n")
            continue

        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def test_one_epoch(model, test_loader, device):
    model.eval()
    total_loss= 0.0
    with torch.no_grad():
        for x_true, params, inputs in test_loader:
            x_true, params, inputs = x_true.to(device), params.to(device), inputs.to(device)
            loss = koopman_loss(model, x_true, params, inputs)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def train(model, optimizer, steplr, train_loader, test_loader, device, epochs, log_file):
    train_losses = []
    test_losses = []

    # Wrap the epochs loop with tqdm for a progress bar
    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, log_file)

        # Test after the epoch
        test_loss = test_one_epoch(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        steplr.step()

        # Update the progress bar with the current epoch's losses
        progress_bar.set_postfix({
            "Train Loss": f"{train_loss:.3e}", 
            "Test Loss": f"{test_loss:.3e}", 
        })

    return train_losses, test_losses,

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read Configurations
    args = parse_arguments()
    config = read_config_file(args.config)
    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the dataset
    data_dir = config["data_dir"]
    step_size = config["step_size"]
    pca_dim = config["pca_dim"]
    if config["batch_size"]:
        batch_size = config["batch_size"]
    else:
        batch_size = 256
    if config["validation_split"]:
        validation_split = config["validation_split"]
    else:
        validation_split = 0.2

    train_loader, test_loader, dataset = get_dataset(data_dir, step_size, pca_dim, batch_size, validation_split)

    torch.save(dataset, os.path.join(save_dir, "dataset.pth"))
    
    # Initialize the model
    for data, params, inputs in train_loader:
        state_dim = data.shape[-1]
        inputs_dim = inputs.shape[-1]
        params_dim = params.shape[-1]
        break
    
    model = ParamOrthogonalKoopmanWithInputs(state_dim, config["dictionary_dim"], inputs_dim, params_dim, config["dictionary_layers"], config["Q_layers"], config["T_layers"], config["B_layers"])
    model.B_matrix.resnet.initialize_weights_to_zero()

    model.to(device)
    

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size_lr"], gamma=config["gamma_lr"])

    log_file = os.path.join(save_dir, "log.txt")
    with open(log_file, "w") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Optimizer: {optimizer}\n")
        f.write(f"StepLR: {steplr}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Config: {config}\n")

    # Train the model
    start_time = time.time()
    train_losses, test_losses = train(model, optimizer, steplr, train_loader, test_loader, device, config["epochs"], log_file)
    end_time = time.time()

    print(f"Training time: {end_time - start_time:.2f}s")
    with open(log_file, "a") as f:
        epoches = config["epochs"]
        f.write(f"Training time for {epoches} epoches is: {end_time - start_time:.2f}s\n")

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state_dict.pth"))

    # Save the losses
    losses = {"train_losses": train_losses, "test_losses": test_losses}
    torch.save(losses, os.path.join(save_dir, "losses.pth"))
    
    return

if __name__ == "__main__":
    main()
        
    