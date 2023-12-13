"""
Evaluates the performance of the 
"""

from riskLogic import Risk, Rules
from boards import ClassicBoard
from players import RiskPlayer, RandomPlayer
from riskNet import RiskNet
from collections import Counter
from constants import *
import matplotlib.pyplot as plt

if TRAINING is True:
    raise ValueError(
        "Training should not be True while evaluating model performance. Please set it to False. Thank you!")

net = RiskNet()

model_name = 'Smart Player'
random_name = 'Random Player'

players = list()
players.append(RiskPlayer(model_name, net))
players.append(RandomPlayer(random_name))
players.append(RandomPlayer(random_name))
players.append(RandomPlayer(random_name))
players.append(RandomPlayer(random_name))
players.append(RandomPlayer(random_name))

board = ClassicBoard(players)
game = Risk(players, Rules(), board)

leader = None
games_won = Counter()
games_won.update({model_name: 0, random_name: 0})

print('\nBeginning evaluation...')
for i in range(EVAL_GAMES):
    if (i + 1) % 10 == 0:
        print(f'Starting game {i + 1}...')
    try:
        leader = game.play()
    except TimeoutError:
        leader = game.get_leader()

    games_won[leader.name] += 1
    board.reset()

plt.bar(list(games_won.keys()), list(games_won.values()))
plt.title('Game Leaders After 10 Turns')
plt.show()
