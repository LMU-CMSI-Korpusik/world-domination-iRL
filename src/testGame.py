"""
A demonstration of the Risk game working using RandomPlayers

                                        ########## IMPORTANT ##########
BEFORE RUNNING THIS, DISABLE THE INCREMENT OF self.board.matches_traded IN Risk.tradein()!!!!!
RandomPlayers are garbage and the game will go on forever if you don't.                                    

Author: Kieran Ahn
Date: 11/27/2023
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from players import RandomPlayer

players = list()
players.append(RandomPlayer('Amogus'))
players.append(RandomPlayer('Morbius'))
players.append(RandomPlayer('sus'))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)
game.play(quiet=False)
print(f'\n\nstate of {players[0].name}:')
print(board.get_state_for_player(players[0]))

print(f'\n\nstate of {players[1].name}:')
print(board.get_state_for_player(players[1]))

print(f'\n\nstate of {players[2].name}:')
print(board.get_state_for_player(players[2]))

board.reset()
