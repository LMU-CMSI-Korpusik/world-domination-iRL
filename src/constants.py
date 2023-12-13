"""
Constants for training the NN.

Author: Kieran Ahn
Date: 12/4/2023
"""

from torch import cuda
import torch
import numpy as np
import random

TRAINING = False
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

seeded = False
seed = 1234

rng = np.random.default_rng(
    seed=seed) if seeded else np.random.default_rng()
if seeded:
    torch.manual_seed(seed)
    random.seed(seed)


# Number of episodes to sample from replay buffer
BATCH_SIZE = 32

# Discount Factor
GAMMA = 0.45

# Soft update rate
TAU = 0.005

# Exploration rate
EPS_START = .99
EPS_END = 0.03
EPS_DECAY = 0.003

# Episode capacity of replay buffer
MEM_SIZE = 10000

# paths for saved data
WEIGHTS_PATH = '../dat/weights_RiskNet.pth'
MEMORY_PATH = '../dat/memory_RiskNet.pkl'

# no game of risk should go for more than ten rounds
MAX_ROUNDS = 10

TRAINING_GAMES = 100000
