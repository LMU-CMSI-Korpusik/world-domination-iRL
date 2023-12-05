"""
Python module containing the RiskNet class, which is a torch.nn.Module subclass
that will play Risk.
"""

import torch
from torch import nn
from riskGame import Action, Design
from validators import *
from constants import *


class RiskNet(nn.Module):
    """
    A neural network that plays Risk.
    """

    def __init__(self, hidden_layer: int = 512, territories: int = 42, players: int = 6):
        super().__init__()
        maximum_potential_cards = 9
        card_designs = len(Design)
        in_features = 751  # This should be calculated from the number of territories and players but I can't get the math right so it's just hard-coded for now -Kieran

        self.hidden_states = nn.Sequential(
            nn.Linear(in_features, hidden_layer, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer, dtype=torch.double),
            nn.ReLU()
        )

        # There are probably better ways to do this, but basically, for actions
        # which require the model to select both a territory and # of dice, I
        # just have it as two heads, and we can mask things if there aren't
        # enough armies to attack. For when you're hcoosing a number of armies,
        # I have it as the percentage of armies to place out of the total given

        # things such as armies and card choices can be scaled; e.g., math.floor(output * n_armies)
        # certain heads which give territories have an additional node to
        # represent the choice to not attack/fortify/etc
        self.attack_head_target = nn.Linear(
            hidden_layer, territories + 1, dtype=torch.double)
        self.attack_head_base = nn.Linear(
            hidden_layer, territories, dtype=torch.double)
        self.attack_head_armies = nn.Sequential(
            nn.Linear(hidden_layer, 3, dtype=torch.double), nn.Softmax(dim=0))
        self.placement_head_territory = nn.Linear(
            hidden_layer, territories, dtype=torch.double)
        self.placement_head_armies = nn.Sequential(
            nn.Linear(hidden_layer, 1, dtype=torch.double), nn.Sigmoid())
        self.claim_head = nn.Linear(
            hidden_layer, territories, dtype=torch.double)
        self.capture_head = nn.Sequential(
            nn.Linear(hidden_layer, 1, dtype=torch.double), nn.Sigmoid())
        # since there are only 2 options, this is basically binary, so we can use the sigmoid
        self.defend_head = nn.Sequential(
            nn.Linear(hidden_layer, 1, dtype=torch.double), nn.Sigmoid())
        self.fortify_head_source = nn.Linear(
            hidden_layer, territories, dtype=torch.double)
        self.fortify_head_destination = nn.Linear(
            hidden_layer, territories + 1, dtype=torch.double)
        self.fortify_head_armies = nn.Sequential(
            nn.Linear(hidden_layer, 1, dtype=torch.double), nn.Sigmoid())
        self.card_choice_head = nn.Linear(
            hidden_layer, 1, dtype=torch.double)

    def forward(self, x, action_taken: Action, selectable_options: list[bool] = None):
        """
        Forward pass of the model

        :params:
        x                   --  the state\n
        action_taken        --  the action the model is performing\n
        selectable_options  --  a list corresponding to the valid territories
        or cards that the model can choose from, where 1 = valid, and 0 =
        invalid
        """
        hidden_state_out = self.hidden_states(x)
        match(action_taken):
            case Action.CHOOSE_ATTACK_TARGET:
                target_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE)
                return target_mask * nn.functional.softmax(self.attack_head_target(hidden_state_out), dim=0, dtype=torch.double)
            case Action.CHOOSE_ATTACK_BASE:
                base_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE
                )
                return base_mask * nn.functional.softmax(self.attack_head_base(hidden_state_out), dim=0, dtype=torch.double), self.attack_head_armies(hidden_state_out)
            case Action.PLACE:
                placement_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE)
                return placement_mask * nn.functional.softmax(self.placement_head_territory(hidden_state_out), dim=0, dtype=torch.double), self.placement_head_armies(hidden_state_out)
            case Action.CLAIM:
                claim_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE)
                return claim_mask * nn.functional.softmax(self.claim_head(hidden_state_out), dim=0, dtype=torch.double)
            case Action.CAPTURE:
                return self.capture_head(hidden_state_out)
            case Action.DEFEND:
                return self.defend_head(hidden_state_out)
            case Action.CHOOSE_FORTIFY_TARGET:
                source_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE)
                return source_mask * nn.functional.softmax(self.fortify_head_destination(hidden_state_out), dim=0, dtype=torch.double)
            case Action.CHOOSE_FORTIFY_SOURCE:
                destination_mask = torch.tensor(
                    selectable_options, dtype=torch.double, device=DEVICE)
                return destination_mask * nn.functional.softmax(self.fortify_head_source(hidden_state_out), dim=0, dtype=torch.double), self.fortify_head_armies(hidden_state_out)
            case Action.CARDS:
                return nn.functional.softmax(self.card_choice_head(hidden_state_out), dim=0, dtype=torch.double)
            case _:
                raise RuntimeError(
                    f'Invalid action supplied. Action given: {action_taken}')
