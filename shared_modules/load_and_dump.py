#
import os
import pickle
import torch
import numpy as np
from types import SimpleNamespace
from datetime import datetime
import DeepExplore

def dump_arrays(array, name, tensor=False):

    # 1. Make path
    results_dir = os.getcwd() + '/results'

    # 2. Save file
    if tensor:
        file_path = results_dir + '/tensors/' + name

        file_path_torch = file_path + '.pt'

        torch.save(array, file_path_torch)

    else:
        file_path = results_dir + '/arrays/' + name

        file_path_np = file_path + '.npy'

        with open(file_path_np, 'wb') as f:
            np.save(f, array)

def load_arrays(array_path, tensor=False, get_cwd=False):

    if get_cwd:
        array_path = os.getcwd() + array_path

    if tensor:
        return torch.load(array_path, map_location=torch.device('cpu'))
    
    else:
        with open(array_path, 'rb') as f:
            container = np.load(f)

        return container

def dump_nn(nn, name):

    file_path = os.getcwd() + '/results/nns/nn_' + name

    torch.save(nn.state_dict(), file_path)

def load_nn(nn_name, neurons, cpu=True):

    # 1. Construct new nn
    loaded_model = DeepExplore.Policy(4, 1, neurons)

    # 2. Load into new nn
    if cpu:
        loaded_model.load_state_dict(torch.load(os.getcwd() + '/results/nns/nn_' + nn_name, weights_only=False, map_location=torch.device('cpu'))) 
    
    else:
        loaded_model.load_state_dict(torch.load(os.getcwd() + '/results/nns/nn_' + nn_name))

    return loaded_model