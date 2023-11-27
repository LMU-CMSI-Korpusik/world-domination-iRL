"""
Different types of Players that can play in Risk

Author: Kieran Ahn
Date: 11/27/2023
"""

from riskGame import Card, Territory
from riskLogic import Board, Player, Rules
import random


class RandomPlayer(Player):
    """
    A very simple Player that just makes random choices in all situations.
    """

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        return random.choice(list(free_territories))

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        return (random.choice(list(self.territories)), random.randint(1, armies_to_place))

    def attack(self, board: Board) -> tuple[Territory, Territory, int]:
        stop_attack = random.random() * (self.occupied_territories() + 1) < 1
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

            target = random.choice(list(valid_attack_targets))
            valid_attack_targets.remove(target)

            valid_bases = self.get_valid_bases(
                board, target, occupied_territories)

            if len(valid_bases) != 0:
                choosing = False

        base = random.choice(list(valid_bases))

        attacking_armies = random.randint(1, min(board.armies[base], 3))

        return target, base, attacking_armies

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        return random.randint(attacking_armies, board.armies[base] - 1)

    def defend(self, board_state: Board, target: Territory) -> int:
        return random.randint(1, min(board_state.armies[target], 2))

    def fortify(self, board: Board) -> tuple[Territory, Territory, int]:
        no_fortify = random.random() * (self.occupied_territories() + 1) < 1

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

            destination = random.choice(possible_destinations)
            possible_destinations.remove(destination)

            valid_sources = self.get_valid_bases(
                board, destination, occupied_territories)

            if len(valid_sources) != 0:
                choosing = False

        source = random.choice(list(valid_sources))

        return destination, source, random.randint(1, board.armies[source] - 1)

    def use_cards(self, board: Board) -> tuple[Card, Card, Card]:
        return random.choice(Rules.get_matching_cards(self.hand))

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        return random.choice(potential_territories)
