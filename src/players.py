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

        choosing = True
        valid_bases = None
        target = None
        while choosing:
            if len(valid_attack_targets) == 0:
                return (None, None, None)

            target = rng.choice(list(valid_attack_targets))
            valid_attack_targets.remove(target)

            valid_bases = self.get_valid_bases(
                board, target, occupied_territories)

            if len(valid_bases) != 0:
                choosing = False

        base = rng.choice(list(valid_bases))

        attacking_armies = None

        attacking_armies = rng.integers(
            1, min(board.armies[base] - 1, 3), endpoint=True)

        return target, base, int(attacking_armies)

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        if board.armies[base] - 1 == attacking_armies:
            return attacking_armies
        return int(rng.integers(attacking_armies, board.armies[base]))

    def defend(self, board_state: Board, target: Territory) -> int:
        if board_state.armies[target] == 1:
            return 1
        else:
            return int(rng.integers(1, 2, endpoint=True))

    def fortify(self, board: Board) -> tuple[Territory, Territory, int]:
        no_fortify = rng.random() * (self.occupied_territories() + 1) < 1

        if no_fortify:
            return (None, None, None)

        occupied_territories = set(list(self.territories))
        possible_destinations = [
            territory for territory in occupied_territories]

        choosing = True
        valid_sources = None
        destination = None
        while choosing:
            if len(possible_destinations) == 0:
                return (None, None, None)

            destination = rng.choice(possible_destinations)
            possible_destinations.remove(destination)

            valid_sources = self.get_valid_bases(
                board, destination, occupied_territories)

            if len(valid_sources) != 0:
                choosing = False

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
        mask = [0 for _ in len(board.territories)]
        for territory in valid_territories:
            mask[board.territory_to_index[territory]] = 1
        return mask

    @staticmethod
    def get_territory_from_index(board: Board, territory_index: int) -> Territory:
        for territory, index in board.territory_to_index:
            if territory_index == index:
                return territory

    def __init__(self, name: str, net: RiskNet):
        self.name = name
        self.net = net
        self.territories = dict()
        self.hand = list()

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        state = board.get_state_for_player(self)
        valid_selections = self.get_territories_mask(board, free_territories)
        claim_index = np.argmax(
            self.net(state, Action.CLAIM, valid_selections))
        claim = self.get_territory_from_index(board, claim_index)

        return claim

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        state = board.get_state_for_player(self)
        valid_selections = self.get_territories_mask(
            board, list(self.territories.keys()))
        territory_activations, armies_proportion = self.net(
            state, Action.PLACE, valid_selections)
        territory = self.get_territory_from_index(
            board, np.argmax(territory_activations))

        return territory, int(np.round(armies_proportion * armies_to_place))

    def attack(self, board: Board) -> tuple[Territory | None, Territory, int]:
        state = board.get_state_for_player(self)
        valid_selections = self.get_territories_mask(
            board, self.get_valid_attack_targets(board, list(self.territories.keys())))
        target_index = np.argmax(
            self.net(state, Action.CHOOSE_ATTTACK_TARGET, valid_selections))

        if target_index == len(board.territories):
            return None, None, None

        target = self.get_territory_from_index(board, target_index)

        valid_selections = self.get_territories_mask(
            board, [territory for territory in board.territories[target] if territory in self.territories])
        base_activations, armies_activations = self.net(
            state, Action.CHOOSE_ATTACK_BASE, valid_selections)
        base = self.get_territory_from_index(np.argmax(base_activations))

        armies_mask = torch.tensor([1.0 if board.armies[base] > potential_armies else 0.0 for potential_armies in [
                                   1, 2, 3]], dtype=torch.float).to(DEVICE)
        armies = np.argmax(armies_mask * armies_activations) + 1
        return target, base, armies

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        pass
