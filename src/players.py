"""
Different types of Players that can play in Risk

Author: Kieran Ahn
Date: 11/27/2023
"""
from validators import *
from riskGame import Card, Territory, Action
from riskLogic import Board, Player
from riskNet import RiskNet
import numpy as np
import torch

rng = np.random.default_rng()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RandomPlayer(Player):
    """
    A very simple Player that just makes random choices in all situations.
    """

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        return rng.choice(list(free_territories))

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        armies_placed = None
        if armies_to_place == 1:
            armies_placed = 1
        else:
            armies_placed = int(rng.integers(
                1, armies_to_place, endpoint=True))
        return (rng.choice(list(self.territories)), armies_placed)

    def attack(self, board: Board) -> tuple[Territory, Territory, int]:
        stop_attack = rng.random() * (self.occupied_territories() + 1) < 1
        occupied_territories = set(list(self.territories))

        if stop_attack:
            return (None, None, None)

        valid_attack_targets = self.get_valid_attack_targets(
            board, occupied_territories)

        if len(valid_attack_targets) == 0:
            return (None, None, None)

        target = rng.choice(list(valid_attack_targets))

        valid_bases = self.get_valid_bases(
            board, target, occupied_territories)

        base = rng.choice(list(valid_bases))

        attacking_armies = rng.integers(
            1, min(board.armies[base] - 1, 3), endpoint=True)

        return target, base, int(attacking_armies)

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        if board.armies[base] - 1 == attacking_armies:
            return attacking_armies
        return int(rng.integers(attacking_armies, board.armies[base]))

    def defend(self, board_state: Board, target: Territory, attacking_armies: int) -> int:
        if board_state.armies[target] == 1:
            return 1
        else:
            return int(rng.integers(1, 2, endpoint=True))

    def fortify(self, board: Board) -> tuple[Territory, Territory, int]:
        no_fortify = rng.random() * (self.occupied_territories() + 1) < 1

        if no_fortify:
            return (None, None, None)

        occupied_territories = set(list(self.territories))
        possible_destinations = [territory for territory, neighbors in board.territories.items()
                                 if territory in self.territories
                                 and len([neighbor for neighbor in neighbors if neighbor in self.territories and board.armies[neighbor] > 1]) != 0]

        if len(possible_destinations) == 0:
            return (None, None, None)

        destination = rng.choice(possible_destinations)
        valid_sources = self.get_valid_bases(
            board, destination, occupied_territories)

        source = rng.choice(list(valid_sources))

        fortifying_armies = None

        if board.armies[source] == 2:
            fortifying_armies = 1
        else:
            fortifying_armies = int(rng.integers(1, board.armies[source]))

        return destination, source, fortifying_armies

    def use_cards(self, board: Board, matches: list[Card]) -> tuple[Card, Card, Card]:
        if len(matches) == 0:
            return None
        return rng.choice(matches)

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        return rng.choice(potential_territories)


class RiskPlayer(Player):
    """
    A player that implements RiskNet for decisionmaking.
    """
    # if more than 4 cards in hand, mask the None option for tradein to 0

    @staticmethod
    def get_territories_mask(board: Board, valid_territories: list[Territory] | set[Territory]) -> list[int]:
        mask = [0 for _ in range(len(board.territories))]
        for territory in valid_territories:
            mask[board.territory_to_index[territory]] = 1
        return mask

    @staticmethod
    def get_territory_from_index(board: Board, territory_index: int) -> Territory:
        for territory, index in board.territory_to_index.items():
            if territory_index == index:
                return territory

    def __init__(self, name: str, net: RiskNet):
        self.name = name
        self.net = net
        self.territories = dict()
        self.hand = list()

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        state = board.get_state_for_player(
            self, Action.CLAIM, 0, *free_territories)
        valid_selections = self.get_territories_mask(board, free_territories)
        claim_index = int(torch.argmax(
            self.net(state, Action.CLAIM, valid_selections)).item())
        claim = self.get_territory_from_index(board, claim_index)

        return claim

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        state = board.get_state_for_player(self, Action.PLACE, armies_to_place)
        valid_selections = self.get_territories_mask(
            board, list(self.territories.keys()))
        territory_activations, armies_proportion = self.net(
            state, Action.PLACE, valid_selections)
        territory = self.get_territory_from_index(
            board, int(torch.argmax(territory_activations).item()))

        return territory, int(torch.floor(armies_proportion * (armies_to_place - 1)).item() + 1)

    def attack(self, board: Board) -> tuple[Territory | None, Territory, int]:
        pre_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_ATTACK_TARGET)
        valid_selections = self.get_territories_mask(
            board, self.get_valid_attack_targets(board, list(self.territories.keys())))
        valid_selections.append(1)

        target_index = int(torch.argmax(
            self.net(pre_selection_state, Action.CHOOSE_ATTACK_TARGET, valid_selections)).item())

        if target_index == len(board.territories):
            return None, None, None

        target = self.get_territory_from_index(board, target_index)

        valid_selections = self.get_territories_mask(
            board, [territory for territory in board.territories[target] if territory in self.territories and board.armies[territory] > 1])
        post_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_ATTACK_BASE, 0, target)
        base_activations, armies_activations = self.net(
            post_selection_state, Action.CHOOSE_ATTACK_BASE, valid_selections)
        base = self.get_territory_from_index(
            board, int(torch.argmax(base_activations).item()))

        armies_mask = torch.tensor([1.0 if board.armies[base] > potential_armies else 0.0 for potential_armies in [
                                   1, 2, 3]], dtype=torch.float).to(DEVICE)
        armies = int(torch.argmax(armies_mask * armies_activations).item()) + 1
        return target, base, armies

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        state = board.get_state_for_player(
            self, Action.CAPTURE, attacking_armies, target, base)

        return int(attacking_armies + torch.floor(self.net(state, Action.CAPTURE) * (board.armies[base] - attacking_armies - 1)).item())

    def defend(self, board: Board, target: Territory, attacking_armies: int) -> int:
        state = board.get_state_for_player(
            self, Action.DEFEND, attacking_armies, target)

        if board.armies[target] == 1:
            return 1

        defend_activation = self.net(state, Action.DEFEND)

        if defend_activation > 0.5:
            return 2
        else:
            return 1

    def fortify(self, board: Board) -> tuple[Territory | None, Territory, int]:
        pre_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_FORTIFY_TARGET)

        valid_destinations = [territory for territory, neighbors in board.territories.items()
                              if territory in self.territories
                              and len([neighbor for neighbor in neighbors if neighbor in self.territories and board.armies[neighbor] > 1]) != 0]

        valid_selections = self.get_territories_mask(
            board, valid_destinations)

        valid_selections.append(1)

        destination_index = int(torch.argmax(self.net(
            pre_selection_state, Action.CHOOSE_FORTIFY_TARGET, valid_selections)).item())

        if destination_index == len(board.territories):
            return None, None, None

        destination = self.get_territory_from_index(board, destination_index)

        post_selection_state = board.get_state_for_player(
            self, Action.CHOOSE_FORTIFY_SOURCE, 0, destination)

        valid_sources = self.get_territories_mask(board, self.get_valid_bases(
            board, destination, set(list(self.territories.keys()))))

        source_index, armies_proportion = self.net(
            post_selection_state, Action.CHOOSE_FORTIFY_SOURCE, valid_sources)

        source = self.get_territory_from_index(
            board, int(torch.argmax(source_index).item()))
        armies = int(torch.floor(1 + armies_proportion *
                     (board.armies[source] - 2)).item())

        return destination, source, armies

    def use_cards(self, board: Board, matches: list[Card]) -> tuple[Card, Card, Card] | None:
        # TODO: fix this to use proportions
        state = board.get_state_for_player(self, Action.CARDS)

        card_proportion = self.net(state, Action.CAPTURE)

        card_choice = None

        if len(self.hand) < 5:
            card_choice = int(torch.floor(
                card_proportion * len(matches)).item())
        else:
            card_choice = int(torch.floor(
                card_proportion * (len(matches) - 1)).item())

        if card_choice == len(matches):
            return None

        return matches[card_choice]

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        state = board.get_state_for_player(self, Action.PLACE, 2)
        valid_selections = self.get_territories_mask(
            board, potential_territories)
        territory_activations, _ = self.net(
            state, Action.PLACE, valid_selections)
        return self.get_territory_from_index(board, int(torch.argmax(territory_activations).item()))
