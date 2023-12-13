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

try:
    game.play(quiet=False)
except TimeoutError:
    print(f'\nleader was: {game.get_leader().name}')
