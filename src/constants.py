"""
Constants for training the NN.

Author: Kieran Ahn
Date: 12/4/2023
"""

from torch import cuda
import torch
import numpy as np
import random

TRAINING = True
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
# Risk is a game about long-term strategy, so the discount factor is going to be relatively low.
GAMMA = 0.8

# Soft update rate
TAU = 0.005

# Exploration rate
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.005

# Episode capacity of replay buffer
MEM_SIZE = 10000

# paths for saved data
WEIGHTS_PATH = '../dat/weights_RiskNet.pth'
MEMORY_PATH = '../dat/memory_RiskNet.pkl'

# no game of risk should go for more than twenty rounds
MAX_ROUNDS = 20

TRAINING_GAMES = 100000
