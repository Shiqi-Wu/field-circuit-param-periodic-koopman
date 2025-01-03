import numpy as np
import re
import argparse

import os
import sys
# Ensure the utils directory is in the module search path
if '__file__' in globals():
    # Script execution: Use __file__ to get the directory
    current_dir = os.path.dirname(__file__)
else:
    # Interactive environment: Use the current working directory
    current_dir = os.getcwd()

sys.path.append(os.path.join(current_dir, '../utils'))

from args_arguments import parse_arguments, read_config_file

def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        time = float(lines[0].split('=')[1])
        data = [float(line.strip()) for line in lines[1:]]
    return time, data

def read_data(config):
    data_dir = config['data_dir']
    pattern = r"sin\(2pi(\d+\.\d+)time\)\+.*?cos\(2pi(\d+\.\d+)time\).*?Mur=(\d+).*?Js=(\d+\.\d+)"
    for item in os.listdir(data_dir):
        full_dir_path = os.path.join(data_dir, item)
        if os.path.isdir(full_dir_path):
            match = re.search(pattern, full_dir_path)
            if match:
                f1 = float(match.group(1))
                f2 = float(match.group(2))
                mur = int(match.group(3))
                js = float(match.group(4))
                print(f"Found folder: {item}, with f1 = {f1}, f2 = {f2}, mur = {mur}, js = {js}")
            else:
                continue

            try:
                file_names = [os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) if f.startswith('Solution_' and f.endswith('.txt'))]
            except FileNotFoundError:
                print(f"Directory {full_dir_path} not found.")
                continue

            file_names.sort(key = lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

            data_dict = {}
            for i, file_name in enumerate(file_names):
                time, data = read_data_from_txt(file_name)
                I_p = 200 * np.sin(2 * np.pi * f1 * time) + 200 * np.cos(2 * np.pi * f2 * time)
                data_dict[i] = {'time': time, 'data': data, 'I_p': I_p, 'mur': mur, 'js': js}
            
            if not os.path.exists(config['save_dir']):
                os.makedirs(config['save_dir'], exist_ok=True)

            file_path = os.path.join(config['save_dir'], f'dataf1{f1}f2{f2}mur{mur}js{js}.npy')

            np.save(file_path, data_dict)
            print(f"Data has been saved as '{file_path}' file.")

if __name__ == '__main__':
    args = parse_arguments()
    config = read_config_file(args.config)
    if not os.path.exists(config['data_dir']):
        raise TypeError(f"Data directory {config['data_dir']} not found.")
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'], exist_ok=True)
    read_data(config)