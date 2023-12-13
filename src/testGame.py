"""
A demonstration of the Risk game working using RandomPlayers

Author: Kieran Ahn
Date: 11/27/2023
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from riskGame import Action
from players import RandomPlayer

players = list()
players.append(RandomPlayer('Amogus'))
players.append(RandomPlayer('Morbius'))
players.append(RandomPlayer('sus'))

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
