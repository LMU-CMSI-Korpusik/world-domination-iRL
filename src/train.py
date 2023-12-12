"""
Main training loop for RiskPlayer. The best Player after each match is selected
to pass on its parameters and memories to the next iteration. Hopefully, this
will let the networks get better over time.

Author: Kieran Ahn
Date: 12/10/2023
"""

from riskLogic import Risk, Rules, PlayerState
from players import RiskPlayer
from boards import ClassicBoard
from riskNet import RiskNet
from constants import *
import pickle as pkl
import time
from os.path import exists

if TRAINING is False:
    raise ValueError(
        "Training is set to False. Please set it to True. Thank you!")

net = RiskNet()
action_memory = None
if (exists(MEMORY_PATH)):
    with open(MEMORY_PATH, 'rb') as memory:
        action_memory = pkl.load(memory)

players = list()
players.append(RiskPlayer('GLaDOS', net, action_memory))
players.append(RiskPlayer('Skynet', net, action_memory))
players.append(RiskPlayer('HAL', net, action_memory))
players.append(RiskPlayer('VIKI', net, action_memory))
players.append(RiskPlayer('Ultron', net, action_memory))
players.append(RiskPlayer('ChatGPT', net, action_memory))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)

for match in range(TRAINING_GAMES):
    leader = None
    print(f'\nBeginning game {match + 1} at {time.strftime("%H:%M:%S")}...')
    game_start = time.time()
    try:
        leader = game.play(quiet=True)
        game_end = time.time()
        print(
            f'Game {match + 1} completed at {time.strftime("%H:%M:%S")}! Time taken: {game_end - game_start}')
    except TimeoutError:
        game_end = time.time()
        print(
            f'Game {match + 1} went to time at {time.strftime("%H:%M:%S")}! Time taken: {game_end - game_start}')
        leader = game.get_leader()

        for player in players:
            player.final(PlayerState(None, None, None, None, None, None,
                         None, None, None, None, None, None, None, player is leader))
        board.reset()
