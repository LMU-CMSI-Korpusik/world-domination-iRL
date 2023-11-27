"""
Different types of Players that can play in Risk

Author: Kieran Ahn
Date: 11/27/2023
"""

from riskGame import Card, Territory
from riskLogic import Board, Player, Rules
import numpy as np

rng = np.random.default_rng()


class RandomPlayer(Player):
    """
    A very simple Player that just makes random choices in all situations.
    """

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        return rng.choice(list(free_territories))

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        return (rng.choice(list(self.territories)), rng.integers(1, armies_to_place, endpoint=True))

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

        attacking_armies = rng.integers(
            1, min(board.armies[base], 3, endpoint=True))

        return target, base, attacking_armies

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        return rng.integers(attacking_armies, board.armies[base])

    def defend(self, board_state: Board, target: Territory) -> int:
        return rng.integers(1, min(board_state.armies[target], 2), endpoint=True)

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

        return destination, source, rng.integers(1, board.armies[source])

    def use_cards(self, board: Board) -> tuple[Card, Card, Card]:
        return rng.choice(Rules.get_matching_cards(self.hand))

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        return rng.choice(potential_territories)
