"""
Python module containing the RiskNet class, which is a torch.nn.Module subclass
that will play Risk.
"""

import torch
from torch import nn
from riskGame import Action
from validators import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RiskNet(nn.Module):
    """
    A neural network that plays Risk.
    """

    def __init__(self, in_features: int, hidden_layer: int = 512, territories: int = 42):
        super().__init__()
        self.hidden_states = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer),
            nn.ReLU()
        )

        # There are probably better ways to do this, but basically, for actions
        # which require the model to select both a territory and # of dice, I
        # just have it as two heads, and we can mask things if there aren't
        # enough armies to attack. For when you're hcoosing a number of armies,
        # I have it as the percentage of armies to place out of the total given

        # things such as armies and card choices can be scaled; e.g., math.floor(output * n_armies)
        self.attack_head_territory = nn.Linear(hidden_layer, territories)
        self.attack_head_armies = nn.Linear(hidden_layer, 3)
        self.placement_head_territory = nn.Linear(hidden_layer, territories)
        self.placement_head_armies = nn.Sigmoid(nn.Linear(hidden_layer, 1))
        self.claim_head = nn.Linear(hidden_layer, territories)
        self.capture_head = nn.Sigmoid(nn.Linear(hidden_layer, 1))
        # since there are only 2 options, this is basically binary, so we can use the sigmoid
        self.defend_head = nn.Sigmoid(nn.Linear(hidden_layer, 1))
        self.fortify_head_source = nn.Linear(hidden_layer, territories)
        self.fortify_head_target = nn.Linear(hidden_layer, territories)
        self.fortify_head_armies = nn.Sigmoid(nn.Linear(hidden_layer, 1))
        self.card_choice_head = nn.Sigmoid(nn.Linear(hidden_layer, 1))

    def forward(self, x, action_taken: Action, territories: list[int] = None):
        hidden_state_out = self.hidden_states(x)
        match(action_taken):
            case Action.ATTACK:
                validate_is_type(territories, list)
                target_mask = torch.tensor(territories, device=DEVICE)
                return (nn.Softmax(target_mask * self.attack_head_territory(hidden_state_out)), self.attack_head_armies(hidden_state_out))
            case Action.PLACE:
                validate_is_type(territories)
                placement_mask = torch.tensor(territories, device=DEVICE)
                return (nn.Softmax(placement_mask * self.placement_head_territory(hidden_state_out)), self.placement_head_armies(hidden_state_out))
            case Action.CLAIM:
                validate_is_type(territories, list)
                claim_mask = torch.tensor(territories, device=DEVICE)
                return nn.Softmax(claim_mask * self.claim_head(hidden_state_out))
            case Action.CAPTURE:
                return self.capture_head(hidden_state_out)
            case Action.DEFEND:
                return self.defend_head(hidden_state_out)
            case Action.CHOOSE_FORTIFY_SOURCE:
                validate_is_type(territories, list)
                source_mask = torch.tensor(territories, device=DEVICE)
                return nn.Softmax(source_mask * self.fortify_head_source(hidden_state_out))
            case Action.CHOOSE_FORTIFY_TARGET:
                validate_is_type(territories)
                destination_mask = torch.tensor(territories, device=DEVICE)
                return nn.Softmax((destination_mask * self.fortify_head_target(hidden_state_out)), self.fortify_head_armies(hidden_state_out))
            case Action.CARDS:
                return self.card_choice_head(hidden_state_out)
            case _:
                raise RuntimeError(
                    f'Invalid action supplied. Action given: {action_taken}')