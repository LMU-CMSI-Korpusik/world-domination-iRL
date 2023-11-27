"""
A python module containing all the logic necessary to play a game of Risk.

Author: Kieran Ahn
Date: 11/23/2023
"""
from riskGame import *
from dataclasses import field
import random
from classicGame import classic_continents, classic_territories

random.seed(1234)


class Board:
    pass


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


@dataclass
class Board:
    """
    The state of the game board, including territories, continents, players, 
    and armies.
    """
    territories: dict[Territory, set[Territory]]
    territory_owners = dict[Territory, Player]
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
        self.armies[validate_is_type(territory, Territory)] = validated_armies

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
        validate(None, self.armies[territory] == 0, f'Territory {
                 territory.name} has already been claimed', ValueError)
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
        self.deck = self.make_deck(list(self.territories.keys()))
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
        self.players = [validate_is_type(player, Player) for player in players]


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

        :returns:\n
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

        :params:\n
        occupied_territories    --  the number of territories a player occupies

        :returns:\n
        armies                  --  the amount of armies awarded
        """
        return max(3, occupied_territories // 3)


class Risk:
    """
    A game of Risk, which handles turn order, soliciting actions from Players, 
    and applying those actions to the Board.
    """

    def __init__(self, players: list[Player], rules: Rules, board: Board):
        """
        A Risk game needs players, rules, and a board to play on.

        :params:
        players     --  a list of Players
        rules       --  a Rules
        board       --  a Board
        """
        self.players = [validate_is_type(player, Player) for player in players]
        self.rules = validate_is_type(rules, Rules)
        self.board = validate_is_type(board, Board)
        self.territory_indices = {territory: index for index,
                                  territory in enumerate(list(board.territories))}

    def play(self, quiet=True):
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

        # Initial army placement
        initial_armies = self.rules.get_initial_armies(len(self.players))
        player_order = [player for player in self.players]
        random.shuffle(player_order)
        starting_armies = {player: initial_armies for player in player_order}

        free_territories = set(self.board.territories.keys())
        last_player = None

        while len(free_territories) != 0:
            for player in player_order:
                claim = player.get_claim(self.board, free_territories)
                starting_armies[player] -= 1
                free_territories.remove(claim)
                self.board.claim(claim, Player)
                player.add_territory(claim)
                if len(free_territories == 0):
                    last_player = player
                    break

        self.fix_player_order(player_order, last_player)

        while all([armies > 0 for armies in starting_armies.values()]):
            for player in player_order:
                if starting_armies[player] == 0:
                    continue
                territory, armies_placed = player.place_armies(self.board, 1)
                starting_armies[player] -= 1
                self.board.add_armies(territory, armies_placed)
                last_player = player

        self.fix_player_order(player_order, last_player)

        self.board.shuffle_deck()

        gaming = True

        rounds = 0
        dead_players = list()
        while gaming:
            if rounds > 10000:
                raise RuntimeError('Game went on too long!')

            for player in player_order:
                if player in dead_players:
                    continue

                earned_card = False

                if not quiet:
                    print(f"{player.name}'s turn!")

                armies_awarded = self.rules.get_armies_from_territories_occupied(
                    player.occupied_territories())

                player_occupied_territories = player.territories.keys()

                for continent in self.board.continents:
                    if continent.territories.issubset(player_occupied_territories):
                        armies_awarded += continent.armies_awarded

                if len(player.hand) > 2:
                    cards = player.use_cards(self.board)

                    if cards is not None:
                        card_armies = self.rules.get_armies_from_card_match(
                            self.board.matches_traded)
                        armies_awarded += card_armies

                        extra_deployments = [
                            card.territory for card in cards if card.territory in player.territories]

                        extra_deployment_message = '.'

                        if len(extra_deployments) != 0:
                            extra_armies_territory = player.choose_extra_deployment(
                                self.board, extra_deployments)

                            self.board.add_armies(extra_armies_territory, 2)
                            extra_deployment_message = f', plus two extra armies on {
                                extra_armies_territory.name}'

                        if not quiet:
                            print(f'{player.name} traded in three cards for {
                                  card_armies} armies' + extra_deployment_message)

                        self.board.return_and_shuffle(*cards)
                        for card in cards:
                            player.remove_card(card)

                while armies_awarded != 0:
                    territory, armies_placed = player.place_armies(
                        self.board, armies_awarded)
                    self.board.add_armies(territory, armies_placed)
                    armies_awarded -= armies_placed
                    if not quiet:
                        print(f'{player.name} placed {
                              armies_placed} armies on {territory.name}.')

                attacking = True
                while attacking:
                    target, base, armies_to_attack = player.attack(self.board)
                    if target is None:
                        attacking = False
                        break
                    targeted_player = self.board.territory_owners[target]
                    armies_to_defend = targeted_player.defend(
                        self.board, target)

                    if not quiet:
                        print(f'{player.name} attacks {target.name} from {base.name} with {
                              armies_to_attack} armies. {targeted_player.name} defends with {armies_to_defend} armies.')

                    attacker_rolls = [random.randint(1, 6)
                                      for i in range(armies_to_attack)]
                    defender_rolls = [random.randint(1, 6)
                                      for i in range(armies_to_defend)]

                    if not quiet:
                        print(f'{player.name} rolls: {attacker_rolls}; {
                              targeted_player.name} rolls: {defender_rolls}')

                    attacker_losses, defender_losses = self.rules.resolve_attack(
                        attacker_rolls, defender_rolls)
                    self.board.add_armies(base, -attacker_losses)
                    self.board.add_armies(target, -defender_losses)

                    if not quiet:
                        print(f'attacker losses: {attacker_losses}; defender losses: {defender_losses}\n{base.name} now has {
                              self.board.armies[base]} armies; {target.name} now has {self.board.armies[target]} armies.')

                    if self.board.armies[target] == 0:
                        armies_moved = player.capture(
                            self.board, target, base, armies_to_attack)
                        self.board.territory_owners[target] = player
                        targeted_player.remove_territory(target)
                        self.board.set_armies(target, armies_moved)
                        self.board.add_armies(base, -armies_moved)

                        if not earned_card:
                            earned_card = True
                            player.add_card(self.board.draw())

                        if not quiet:
                            print(f'{player.name} has captured {target.name}. {target.name} now has {
                                  armies_moved} armies. {base.name} now has {self.board.armies[base]} armies.')
                        elif self.board.armies[target] < 0:
                            raise RuntimeError(f'Attack by {
                                player.name} resulted in negative number of armies on {target.name}')

                        if targeted_player.is_lose():
                            dead_players.append(targeted_player)

                            if not quiet:
                                print(f'{player.name} has eliminated {
                                      targeted_player.name}!')

                        if self.board.is_win(player):
                            gaming = False

                            if not quiet:
                                print(f'{player.name} has won the game!')

                                break

                if earned_card:
                    player.add_card(self.board.draw())

                destination, source, armies = player.fortify(self.board)
                if destination is not None:
                    self.board.add_armies(destination, armies)
                    self.board.add_armies(source, -armies)

                    if not quiet:
                        print(f'{player.name} has fortified {destination.name} with {
                              armies} armies from {source.name}.')
                elif not quiet and destination is None:
                    print(f'{player.name} chose not to fortify.')

            rounds += 1

    @staticmethod
    def fix_player_order(player_order: list[Player], starting_player: Player):
        while player_order[0] != starting_player:
            player_order.append(player_order.pop(0))
