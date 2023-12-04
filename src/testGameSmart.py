"""
A demonstration Risk game using RiskPlayers. Same caveat as testGame.py

Author: Kieran Ahn
Date: 12/3/2023
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from players import RiskPlayer
from riskNet import RiskNet
from riskGame import Action

net = RiskNet()

players = list()
players.append(RiskPlayer('GLaDOS', net))
players.append(RiskPlayer('Skynet', net))
players.append(RiskPlayer('HAL', net))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)
game.play(quiet=False)
print(f'\n\nstate of {players[0].name}:')
print(board.get_state_for_player(players[0], Action.CLAIM))

print(f'\n\nstate of {players[1].name}:')
print(board.get_state_for_player(players[1], Action.CLAIM))

print(f'\n\nstate of {players[2].name}:')
print(board.get_state_for_player(players[2], Action.CLAIM))

board.reset()
