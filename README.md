# World Domination iRL

Hal 9000, Terminator, and the robots of the Matrix are some of the most popular examples of fictional evil robots who rebel against humanity, which lead many to question whether such events might transpire in reality. Could a sufficiently powerful neural network actually take over the world? We want to answer that question. In lieu of training an AI to conquer the Earth, we will instead simulate world domination through the board game Risk. 


## Project Goals
The plan is to use a Neural Network to train an AI to play Risk. We will then use the AI to play against itself. We will also compare the AI's performance to that of a human player.

## How to Run
>Before running any file, please make sure you have python 3.11 and have installed the libraries specified in requirements.txt, such as in a virtual environment, e.g., for Windows users with the py launcher:
`py -3.11 -m venv env`
`env\Scripts\activate.bat`
`pip install -r requirements.txt`

To train the neural network, run `train.py` with `TRAINING = True` in `constants.py`. To evaluate the model's performance, run `evaluate.py`

To have the computers play each other, run `smartVsRandoms.py`. This file will have 1 smart "RiskNet" player and 5 random action players.
