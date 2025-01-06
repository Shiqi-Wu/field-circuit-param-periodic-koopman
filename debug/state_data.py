import torch
import numpy as np
from src.args import parse_arguments, read_config_file
import os

device = torch.device("cpu")

# Read Configurations
args = parse_arguments()
config = read_config_file(args.config)
save_dir = config["save_dir"]

# Read the data
data_dir = config["data_dir"]
data_list, params_list, inputs_list = [], [], []
for file in os.listdir(data_dir):
    if file.endswith('.npy'):
        ff = np.load(os.path.join(data_dir, file), allow_pickle=True)
        ff = ff.item()

        if np.isnan(ff['data']).any():
            print(file)
        else:
            print("No NaNs in", file)