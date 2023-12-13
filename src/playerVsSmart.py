"""
Game of 1 Player vs Smart Player

Author: Jason Douglas
Date: 12/12/2023
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from players import RiskPlayer, RandomPlayer, HumanPlayer
from riskNet import RiskNet
from riskGame import Action
from constants import *

net = RiskNet()

players = list()
players.append(RiskPlayer('GLaDOS', net))
players.append(RandomPlayer('amogus'))
players.append(RandomPlayer('sus'))
players.append(RandomPlayer('morbius'))
players.append(RandomPlayer('bingus'))
players.append(HumanPlayer('Kieran'))


board = ClassicBoard(players)
game = Risk(players, Rules(), board)

try:
    game.play(quiet=False)
except TimeoutError:
    print(f'\nleader was: {game.get_leader().name}')
    smart_state = board.get_state_for_player(
        players[0], Action.CHOOSE_FORTIFY_ARMIES)
    print('\nSmart player territories:')
    print(smart_state.owned_territories)
    print('\nArmies on territories:')
    print(smart_state.armies)
    print(players[0].network(smart_state.get_features(),
          Action.CHOOSE_FORTIFY_ARMIES))
    smart_state = board.get_state_for_player(
        players[0], Action.CAPTURE)
    print(players[0].network(smart_state.get_features(),
          Action.CAPTURE))