import itertools

import random

import numpy as np
import torch


def setup_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def split_data(x_train, y_train, split_size=0.8):
    num_training_examples = int(split_size * x_train.shape[0])
    x_train, x_val = x_train.iloc[:num_training_examples, :], x_train.iloc[num_training_examples:, :]
    y_train, y_val = y_train.iloc[:num_training_examples, :], y_train.iloc[num_training_examples:, :]
    return x_train, x_val, y_train, y_val


def generate_params_for_grid_search(param_grid):
    return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
