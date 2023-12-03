from riskGame import *
from players import Player
from dataclasses import dataclass, field
from validators import *
from classicGame import classic_territories, classic_continents
import random


@dataclass
class Board:
    """
    The state of the game board, including territories, continents, players, 
    and armies.
    """
    territories: dict[Territory, set[Territory]]
    armies: dict[Territory, int]
    continents: set[Continent]
    players: list[Player]
    deck: list[Card]
    territory_owners: dict[Territory, Player] = field(default_factory=dict)
    matches_traded: int = 0
    territory_to_index: dict[Territory, int] = field(default_factory=dict)

    def is_win(self, player: Player):
        """
        Method to determine whether a player has won the game.

        :params:\n
        player  --  a Player
        """
        if len(self.territories) - len(player.territories) == 0:
            return True
        return False

    def add_armies(self, territory: Territory, armies: int):
        """
        Adds a number of armies to a Territory

        :params:\n
        territory   --  a Territory
        armies      --  the number of armies to add to the Territory
        """
        validated_armies = validate_is_type(armies, int)
        self.armies[validate_is_type(territory, Territory)] += validated_armies

    def set_armies(self, territory: Territory, armies: int):
        """
        Updates the amount of armies on a Territory

        :params:\n
        territory   --  a Territory\n
        armies      --  the new amount of armies on Territory
        """
        validated_armies = validate_is_type(armies, int)
        validated_armies = validate(
            validated_armies, validated_armies > 0, 'Territories must hold at least 1 army', ValueError)
        self.armies[validate_is_type(territory, Territory)] = validated_armies

    def claim(self, territory: Territory, player: Player):
        """
        Claims a territory, setting its armies to 1.

        :params:\n
        territory   --  a Territory
        player      --  the Player who claimed the Territory
        """
        validate_is_type(territory, Territory)
        validate(None, territory in self.territories,
                 'Cannot claim a nonexistent territory', ValueError)
        validate(None, self.armies[territory] == 0,
                 f'Territory {territory.name} has already been claimed', ValueError)
        self.armies[territory] = 1
        self.territory_owners[territory] = player

    def draw(self) -> Card:
        """
        Draws a card from the deck

        :returns:\n
        card    --  the drawn card
        """
        return self.deck.pop()

    def return_and_shuffle(self, *cards: Card):
        """
        Shuffles a number of cards back into the deck

        :params:\n
        cards   --  the cards to return to the deck
        """
        for card in cards:
            self.deck.append(card)

        random.shuffle(self.deck)

    def shuffle_deck(self):
        """
        Shuffles the deck
        """
        random.shuffle(self.deck)

    @staticmethod
    def make_deck(territories: list[Territory]):
        """
        Creates a deck of cards from a list of territories, plus two wildcards.

        :params:\n
        territories -- a list of Territories

        :returns:\n
        deck        -- a list of Cards
        """
        deck = list()
        designs = list(Design)
        for territory in territories:
            deck.append(Card(territory, random.choice(designs)))
        deck.append(Card(None, None, True))
        deck.append(Card(None, None, True))

        return deck

    def reset(self):
        self.armies = {territory: 0 for territory in list(self.territories)}
        for player in self.players:
            player_cards = player.hand
            self.return_and_shuffle(*player_cards)
            player.remove_cards(*player_cards)
        self.matches_traded = 0


class ClassicBoard(Board):
    """
    A classic Risk board, with 42 territories, one card for each territory,
    and two wildcards, organized into 6 continents
    """

    def __init__(self, players: list[Player]):
        """
        The classic board configuration is set, and the only thing that needs
        to be supplied is the players.

        :params:\n
        players     --  a list of Players
        """
        self.territories = classic_territories
        self.continents = classic_continents
        self.armies = {territory: 0 for territory in list(self.territories)}
        self.deck = self.make_deck(list(classic_territories.keys()))
        self.matches_traded = 0
        self.territory_owners = dict()
        self.players = [validate_is_type(player, Player) for player in players]
        self.territory_to_index = {
            territory: index for index, territory in enumerate(list(self.territories))}
