"""
A python module containing all of the necessary data structures to play a game
of Risk.

Author: Kieran Ahn
Date: 11/23/2023
"""
from validators import *
from dataclasses import dataclass
from typing import Self
from enum import Enum


class Design(Enum):
    """
    Designs that can appear on cards.
    """
    INFANTRY = 0
    CAVALRY = 1
    ARTILLERY = 2


@dataclass
class Territory:
    """
    The basic piece of land that one can acquire in Risk. Territories are 
    connected to other territories, and players fight for territories to win 
    the game.
    """
    name: str
    neighbors: set[Self]

    def __init__(self, name: str, neighbors: set[Self] = None):
        """
        A territory must have a name (e.g., Yugoslavia) and neighbors. A 
        territory can be initialized without any neighbors, but that means
        it will be inaccessible in the game world, like Madagascar in 
        Plague, Inc. after a single person sneezes halfway across the world.

        :params:\n
        name        --  the name of the Territory\n
        neighbors   --  a list of other Territories
        """
        self.name = name
        self.neighbors = [validate_is_type(neighbor, Territory) for neighbor in validate_is_type(
            neighbors, list)] if neighbors else set()

    def add_neighbor(self, neighbor: Self):
        """
        Adds a neighbor to the Territory

        :params:\n
        neighbor    --  a Territory
        """
        self.neighbors.add(validate_is_type(neighbor, Territory))


@dataclass
class Continent:
    """
    A group of Territories that, when controlled, awards you a number of armies 
    at the beginning of your turn.
    """
    name: str
    territories: set[Territory]
    armies_awarded: int


@dataclass
class Card:
    """
    Cards are awarded after you complete a turn in which you have captured a 
    territory. Sets of three can be traded for armies at the beginning of a
    player's turn. If you have five or six cards in your hand at the beginning
    of your turn, you MUST trade in at least one set. The number of armies
    awarded for trading in a set of cards increases the more sets are traded
    in. If any card contains a Territory you occupy, you are awarded
    two extra armies which you must place onto that Territory. You cannot
    receive more than two extra armies in this way. If you defeat a player,
    you get all of their cards.
    """
    territory: Territory
    design: Design
    wildcard: bool = False