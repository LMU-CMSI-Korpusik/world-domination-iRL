"""
Constants for training the NN.

Author: Kieran Ahn
Date: 12/4/2023
"""

from torch import cuda
import numpy as np

TRAINING = True
# DEVICE = 'cuda' if cuda.is_available() else 'cpu'
DEVICE = 'cpu'

rng = np.random.default_rng()

# Number of episodes to sample from replay buffer
BATCH_SIZE = 32

# Discount Factor
GAMMA = 0.85

# Exploration rate
EPS_START = 0.951
EPS_END = 0.01
EPS_DECAY = 0.001

# Number of episodes between when the target network's weights are set to the policy net's weights
TARGET_UPDATE = 200

# Episode capacity of replay buffer
MEM_SIZE = 10000

# paths for saved data
WEIGHTS_PATH = '../dat/weights_RiskNet.pth'
MEMORY_PATH = '../dat/memory_RiskNet.pkl'

# no game of risk should go for more than a thousand rounds
MAX_ROUNDS = 1000
