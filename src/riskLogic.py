"""
A python module containing all the logic necessary to play a game of Risk.

Author: Kieran Ahn
Date: 11/23/2023
"""
from riskGame import *
from random import shuffle

"""
PSEUDOCODE FOR RUNNING THE GAME:
    initialize game board from settings (# of players, territories)
    assign armies to players
    get player turn order (list)
    initial army placement:
        for player in turn order if there are unclaimed territories:
            assign army to unclaimed territory
        for player in turn order while players have armies:
            assign army to any territory owned by player
    
    shuffle cards
        
    while over is not True:
        for player in turn order:
            player.getNewArmies()
            player.placeArmies()
            player.attack()
            player.fortify()
            eliminate dead players
            if player has won:
                over = True
                break
"""


@dataclass
class Player:
    """
    An agent who will play Risk. Interface to be implemented.
    """
    name: str
    territories: set[Territory] = set()
    hand: set[Card] = set()

    def choose_action():
        raise NotImplementedError(
            "Cannot call choose_action from base player class")

    def add_territory(self, territory: Territory):
        """
        Adds a Territory to the player's Territories, for when a territory is
        captured during play or at the beginning of the game.

        :params:\n
        territory   -- A Territory
        """
        self.territories.add(validate_is_type(territory, Territory))

    def remove_territory(self, territory: Territory):
        """
        Removes a Territory from the players Territories, i.e., when it is 
        captured.

        :params:\n
        territory   -- a Territory
        """
        validated_territory = validate_is_type(territory, Territory)
        if validated_territory in self.territories:
            self.territories.remove(validated_territory)
        else:
            raise ValueError(
                f'Player {self.name} does not own {validated_territory.name}'
            )

    def add_card(self, card: Card):
        """
        Adds a Card to the player's hand, either from drawing or eliminating 
        another player.

        :params:\n
        card    -- a Card
        """
        self.hand.add(validate_is_type(card, Card))

    def remove_card(self, card: Card):
        """
        Removes a Card from a player's hand, such as when turning cards in for
        armies.

        :params:\n
        card    -- a Card
        """
        validated_card = validate_is_type(card, Card)
        if validated_card in self.hand:
            self.hand.remove(validated_card)
        else:
            raise ValueError(
                f'Player {self.name} does not have card {card} in their hand.'
            )


@dataclass
class Board:
    """
    The state of the game board, including territories, continents, players, 
    and armies.
    """
    territories: set[Territory]
    armies: dict[Territory, int]
    continents: set[Continent]
    players: list[Player]
    deck: list[Card]

    def is_win(self, player: Player):
        """
        Method to determine whether a player has won the game.

        :params:\n
        player  -- a Player
        """
        if len(self.territories.difference(validate_is_type(player, Player).territories)) == 0:
            return True
        return False

    def set_armies(self, territory: Territory, armies: int):
        """
        Updates the amount of armies on a territory

        :params:\n
        territory   -- a Territory\n
        armies      -- the new amount of armies on territory
        """
        validated_armies = validate_is_type(armies, int)
        validated_armies = validate(
            validated_armies, validated_armies > 0, 'Territories must hold at least 1 army', ValueError)
        self.armies[validate_is_type(territory, Territory)] = validated_armies

    def draw(self) -> Card:
        """
        Draws a card from the deck

        :returns:\n
        card    -- the drawn card
        """
        return self.deck.pop()

    def return_and_shuffle(self, *cards: Card):
        """
        Shuffles a number of cards back into the deck

        :params:
        cards   -- the cards to return to the deck
        """

        for card in cards:
            self.deck.append(card)

        shuffle(self.deck)
