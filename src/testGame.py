"""
A demonstration of the Risk game working using RandomPlayers

                                        ########## IMPORTANT ##########
BEFORE RUNNING THIS, DISABLE THE INCREMENT OF self.board.matches_traded IN Risk.play()!!!!!
RandomPlayers are garbage and the game will go on forever if you don't.                                    

Author: Kieran Ahn
Date: 11/27/2023
"""

from riskLogic import Risk, ClassicBoard, Rules
from players import RandomPlayer

players = list()
players.append(RandomPlayer('Amogus'))
players.append(RandomPlayer('Morbius'))
players.append(RandomPlayer('sus'))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)
game.play(quiet=False)
