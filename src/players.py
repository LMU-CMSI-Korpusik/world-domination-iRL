"""
Different types of Players that can play in Risk

Author: Kieran Ahn
Date: 11/27/2023
"""
from validators import *
from dataclasses import dataclass, field
from riskGame import Card, Territory
from riskLogic import Rules
from boards import Board
import numpy as np

rng = np.random.default_rng()


@dataclass
class Player:
    """
    An agent who will play Risk. Interface to be implemented.

    :fields:\n
    name        --  the Player's name\n
    territories --  a dict of the Territories owned by the player and their
    corresponding indices in the sparse row\n
    hand        --  the Player's hand of Cards
    """
    name: str
    territories: dict[Territory, int] = field(default_factory=dict)
    hand: list[Card] = field(default_factory=list)

    def get_claim(self, board: Board, free_territories: set[Territory]) -> Territory:
        """
        TODO: document this
        """
        raise NotImplementedError(
            "Cannot call get_claim from base Player class")

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        """
        TODO: document this
        """
        raise NotImplementedError(
            "Cannot call place_armies on base Player class")

    def attack(self, board: Board) -> tuple[Territory, Territory, int]:
        """
        TODO: document this
        """
        raise NotImplementedError("Cannot call attack on base Player class")

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        """
        TODO: document this
        """
        raise NotImplementedError("Cannot call capture on base Player classs")

    def defend(self, board: Board, target: Territory) -> int:
        """
        TODO: document this
        """
        raise NotImplementedError("Cannot call defend on base Player class")

    def fortify(self, board: Board) -> tuple[Territory, Territory, int]:
        """
        TODO: document this
        """
        raise NotImplementedError("Cannot call fortify on base Player class")

    def use_cards(self, board: Board) -> tuple[Card, Card, Card]:
        """
        TODO: document this
        """
        raise NotImplementedError("Cannot call use_cards on base Player class")

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        """
        TODO: document this
        """
        raise NotImplementedError(
            "Cannot call choose_extra_deployment on base Player class")

    def add_territory(self, territory: Territory, territory_index: int):
        """
        Adds a Territory to the player's Territories, for when a territory is
        captured during play or at the beginning of the game.

        :params:\n
        territory       --  a Territory
        territory_index --  the Territory's index in the sparse row
        """
        self.territories.update(
            {validate_is_type(territory, Territory): territory_index})

    def remove_territory(self, territory: Territory):
        """
        Removes a Territory from the players Territories, i.e., when it is 
        captured.

        :params:\n
        territory   --  a Territory
        """
        validated_territory = validate_is_type(territory, Territory)
        if validated_territory in self.territories:
            del self.territories[validated_territory]
        else:
            raise ValueError(
                f'Player {self.name} does not own {validated_territory.name}'
            )

    def add_cards(self, *cards: Card):
        """
        Adds a Card to the player's hand, either from drawing or eliminating 
        another player.

        :params:\n
        card    --  a Card
        """
        for card in cards:
            self.hand.append(validate_is_type(card, Card))

    def remove_cards(self, *cards: Card):
        """
        Removes a Card from a player's hand, such as when turning cards in for
        armies.

        :params:\n
        card    --  a Card
        """
        for card in cards:
            validated_card = validate_is_type(card, Card)
            if validated_card in self.hand:
                self.hand.remove(validated_card)
            else:
                raise ValueError(
                    f'Player {self.name} does not have card {card} in their hand.'
                )

    @staticmethod
    def get_valid_attack_targets(board: Board, occupied_territories: set[Territory]) -> set:
        """
        TODO: document this
        """
        return {neighbor for territory in occupied_territories
                for neighbor in board.territories[territory]
                if neighbor not in occupied_territories}

    @staticmethod
    def get_valid_bases(board: Board, target: Territory, occupied_territories: set[Territory]) -> set:
        """
        TODO: document this
        """
        return {neighbor for neighbor in board.territories[target]
                if neighbor in occupied_territories and board.armies[neighbor] > 2}

    def occupied_territories(self):
        return len(self.territories)

    def is_lose(self):
        return self.occupied_territories() == 0


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

    def use_cards(self, board: Board) -> tuple[Card, Card, Card]:
        matches = Rules.get_matching_cards(self.hand)
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

    pass
