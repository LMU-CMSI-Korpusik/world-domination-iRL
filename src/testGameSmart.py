"""
A demonstration Risk game using RiskPlayers.

Author: Kieran Ahn
Date: 12/3/2023
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from players import RiskPlayer
from riskNet import RiskNet
from riskGame import Action
from constants import *

net = RiskNet()

players = list()
players.append(RiskPlayer('GLaDOS', net))
players.append(RiskPlayer('Skynet', net))
players.append(RiskPlayer('HAL', net))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)

try:
    game.play(quiet=False)
except TimeoutError:
    print(f'\nleader was: {game.get_leader().name}')
