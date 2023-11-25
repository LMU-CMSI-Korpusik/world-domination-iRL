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
    hand: list[Card] = list()

    def choose_action():
        raise NotImplementedError(
            "Cannot call choose_action from base player class")

    def add_territory(self, territory: Territory):
        """
        Adds a Territory to the player's Territories, for when a territory is
        captured during play or at the beginning of the game.

        :params:\n
        territory   --  a Territory
        """
        self.territories.add(validate_is_type(territory, Territory))

    def remove_territory(self, territory: Territory):
        """
        Removes a Territory from the players Territories, i.e., when it is 
        captured.

        :params:\n
        territory   --  a Territory
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
        card    --  a Card
        """
        self.hand.append(validate_is_type(card, Card))

    def remove_card(self, card: Card):
        """
        Removes a Card from a player's hand, such as when turning cards in for
        armies.

        :params:\n
        card    --  a Card
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
    matches_traded: int = 0

    def is_win(self, player: Player):
        """
        Method to determine whether a player has won the game.

        :params:\n
        player  --  a Player
        """
        if len(self.territories.difference(validate_is_type(player, Player).territories)) == 0:
            return True
        return False

    def set_armies(self, territory: Territory, armies: int):
        """
        Updates the amount of armies on a territory

        :params:\n
        territory   --  a Territory\n
        armies      --  the new amount of armies on territory
        """
        validated_armies = validate_is_type(armies, int)
        validated_armies = validate(
            validated_armies, validated_armies > 0, 'Territories must hold at least 1 army', ValueError)
        self.armies[validate_is_type(territory, Territory)] = validated_armies

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

        shuffle(self.deck)


class Rules:
    """
    The rules for running a game of Risk
    """

    @staticmethod
    def get_matching_cards(cards: list[Card]) -> list[tuple[Card, Card, Card]]:
        """
        Finds all the valid matches in a hand of cards

        :params:\n
        cards   --  A list of Cards

        :returns:\n
        matches --  A list of tuples containing matched Cards
        """
        matches = list()

        for i in range(len(cards)):
            for j in range(i, len(cards)):
                if i == j:
                    continue
                # Yes, having a triple for loop hurts my eyes, too.
                for k in range(j, len(cards)):
                    if j == k:
                        continue
                    if (cards[i].design == cards[j].design and cards[j].design == cards[k].design) or (cards[i].design != cards[j].design and cards[j].design != cards[k].design and cards[k].design != cards[i].design) or any([cards[i].wildcard, cards[j].wildcard, cards[k].wildcard]):
                        # Don't talk to me about this conditional. -Kieran
                        matches.append((cards[i], cards[j], cards[k]))

        return matches

    @staticmethod
    def get_initial_armies(n_players: int) -> int:
        """
        Gets the initial number of armies a player starts the game with.

        :params:\n
        n_players   --  the number of players in the game

        :returns:\n
        armies      --  the initial armies for each player
        """
        if n_players < 1:
            raise ValueError(
                f'Please tell me how playing Risk with {
                    n_players} players is mathematically possible.'
            )
        if n_players == 1:
            raise ValueError("You cannot play Risk by yourself.")
        if n_players == 2:
            raise NotImplementedError(
                "2 player Risk has not been implemented yet.")
        if n_players > 6:
            raise ValueError("Risk does not support more than six players.")
        else:
            return 35 - (5 * (n_players - 3))

    @staticmethod
    def resolve_attack(attacker_rolls: list[int], defender_rolls: list[int]) -> tuple[int, int]:
        """
        Determines whether the attacker or defender wins when attacking a
        Territory

        :params:\n
        attacker_rolls  --  The dice rolls the attacker made\n
        defender_rolls  --  The dice rolls the defender made

        :returns:
        attacker_losses --  the number of armies the attacker lost\n
        defender_losses --  the number of armies the defender lost
        """
        attacker_rolls.sort()
        defender_rolls.sort()
        attacker_losses = 0
        defender_losses = 0

        while len(attacker_rolls) > 0 and len(defender_rolls) > 0:
            highest_attacker_roll = attacker_rolls.pop()
            highest_defender_roll = defender_rolls.pop()
            if highest_attacker_roll > highest_defender_roll:
                defender_losses += 1
            else:
                attacker_losses += 1

        return attacker_losses, defender_losses

    @staticmethod
    def get_armies_from_card_match(matches_made: int) -> int:
        """
        The amount of armies turning in a matched set of cards will get you

        :params:\n
        matches_made    --  the number of matched trios turned in so far

        :returns:\n
        armies          --  the amount of armies awarded
        """
        if matches_made < 5:
            return 4 + 2 * matches_made
        elif matches_made == 5:
            return 15
        else:
            return (matches_made - 2) * 5

    @staticmethod
    def get_armies_from_territories_occupied(occupied_territories: int) -> int:
        """
        The amount of armies awarded from occupying territories

        :params:
        occupied_territories    --  the number of territories a player occupies

        :returns:\n
        armies                  --  the amount of armies awarded
        """
        return max(3, occupied_territories // 3)
