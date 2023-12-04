"""
A python module containing all the logic necessary to play a game of Risk.

Author: Kieran Ahn
Date: 11/23/2023
"""
from riskGame import *
import random
from dataclasses import dataclass, field
import torch

random.seed(1234)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

action_to_state_index = {action: index for index, action in enumerate(Action)}


class Board:
    pass


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
                f'Please tell me how playing Risk with {n_players} players is mathematically possible.'
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
        Ask the Player which Territory they would like to claim from the
        remaining free Territories at the beginning of the game

        :params:\n
        board               --  the game Board\n
        free_territories    --  the remaining unclaimed Territories

        :returns:\n
        claim               --  the Player's desired claim
        """
        raise NotImplementedError(
            "Cannot call get_claim from base Player class")

    def place_armies(self, board: Board, armies_to_place: int) -> tuple[Territory, int]:
        """
        Ask the Player which Territory on which they want to place their armies
        during the placement phase

        :params:\n
        board           --  the game Board\n
        armies_to_place --  the total number of armies the player can place

        :returns:\n
        territory       --  the Territory on which they are placing the armies\n
        armies          --  the number of armies they want to place
        """
        raise NotImplementedError(
            "Cannot call place_armies on base Player class")

    def attack(self, board: Board) -> tuple[Territory | None, Territory, int]:
        """
        Ask the Player which Territory they would like to attack on their turn

        :params:\n
        board       --  the game Board

        :returns:\n
        target      --  either the Territory the Player wishes to attack or None
        if they do not want to attack at all\n
        base        --  the Territory the Player wishes to use to attack the
        target\n
        armies      --  the number of armies the Player will use for the attack
        """
        raise NotImplementedError("Cannot call attack on base Player class")

    def capture(self, board: Board, target: Territory, base: Territory, attacking_armies: int) -> int:
        """
        When the Player captures a Territory, it must move a number of armies
        to the target from the base. It must move at least as many armies that
        survived from the attack.

        :params:\n
        board               --  the game Board\n
        target              --  the Territory the Player attacked\n
        base                --  the Territory from which the Player attacked\n
        attacking_armies    --  the number of armies the Player used to attack\n

        :returns:\n
        armies              --  the number of armies to move into the captured
        Territory
        """
        raise NotImplementedError("Cannot call capture on base Player classs")

    def defend(self, board: Board, target: Territory, attacking_armies: int) -> int:
        """
        Asks the player whether to defend a Territory with one or two armies

        :params:\n
        board               --  the game Board\n
        target              --  the Territory which is being attacked\n
        attacking_armies    --  the number of armies the attacker is using

        :returns:\n
        armies      --  the number of armies with which to defend the Territory
        """
        raise NotImplementedError("Cannot call defend on base Player class")

    def fortify(self, board: Board) -> tuple[Territory | None, Territory, int]:
        """
        Asks the Player which Territory they would like to fortify at the end
        of their turn

        :params:\n
        board       --  the game Board

        :returns:\n
        destination --  the Territory to fortify, or None if the Player does
        not want to fortify\n
        source      --  the Territory to supply the fortifying armies\n
        armies      --  the number of armies to fortify with
        """
        raise NotImplementedError("Cannot call fortify on base Player class")

    def use_cards(self, board: Board, matches: list[Card]) -> tuple[Card, Card, Card] | None:
        """
        Asks the Player which Cards they would like to turn in at the
        beginning of their turn, if any

        :params:\n
        board       --  the game Board\n
        matches     --  all possible matches the Player can turn in 

        :returns:\n
        cards       --  the trio of Cards to turn in or None if the Player does
        not want to turn in any Cards. Cannot be None if the Player has 5 or 6
        Cards in their hand.
        """
        raise NotImplementedError("Cannot call use_cards on base Player class")

    def choose_extra_deployment(self, board: Board, potential_territories: list[Territory]) -> Territory:
        """
        Asks the Player which Territory they would like to deploy extra armies
        to in the event that they turn in a card with one of their Territories
        on it

        :params:\n
        board                   --  the game Board\n
        potential_territories   --  the Territories available for extra
        deployents

        :returns:\n
        territory               --  the Territory on which the Player would
        like to deploy extra armies
        """
        raise NotImplementedError(
            "Cannot call choose_extra_deployment on base Player class")

    def add_territory(self, territory: Territory, territory_index: int):
        """
        Adds a Territory to the player's Territories, for when a territory is
        captured during play or at the beginning of the game.

        :params:\n
        territory       --  a Territory\n
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
        Adds one or more Cards to the player's hand, either from drawing or 
        eliminating another player.

        :params:\n
        cards    --  one or more Cards
        """
        for card in cards:
            self.hand.append(validate_is_type(card, Card))

    def remove_cards(self, *cards: Card):
        """
        Removes one or more Cards from a player's hand, such as when turning in
        cards for armies.

        :params:\n
        cards    --  one or more Cards
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
        Gets the Territories that a Player can legally attack

        :params:\n
        board                   --  the game Board\n
        occupied_territories    --  the Territories owned by the Player

        :returns:
        valid_targets           --  all valid attack targets for the Player
        """
        return {neighbor for territory in occupied_territories
                for neighbor in board.territories[territory]
                if neighbor not in occupied_territories
                and board.armies[territory] > 1}

    @staticmethod
    def get_valid_bases(board: Board, target: Territory, occupied_territories: set[Territory]) -> set:
        """
        Gets the Territories that the Player can legally use as a source for 
        armies when fortifying a given Territory

        :params:\n
        board                   --  the game Board
        target                  --  the destination for the fortifying armies
        occupied_territories    --  the Territories owned by the Player
        """
        return {neighbor for neighbor in board.territories[target]
                if neighbor in occupied_territories and board.armies[neighbor] > 1}

    def occupied_territories(self):
        return len(self.territories)

    def is_lose(self):
        return self.occupied_territories() == 0


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
            player_territories = list(player.territories.keys())
            for territory in player_territories:
                player.remove_territory(territory)
        self.matches_traded = 0

    def get_state_for_player(self, player: Player, action: Action, armies_used: int = 0, *selected_territories: Territory):
        """
        Returns the state from the view of the supplied player

        :params:\n
        player                  --  The Player for which the state is being
        generated\n
        action                  --  The action with which the Player is being 
        prompted\n
        armies_used             --  The number of armies used to attack\n
        selected_territories    --  Any relevant territories, such as targets

        :returns\n
        state       --  The state
        """
        territories_state = torch.zeros(
            6 * len(self.territories), dtype=torch.float)
        for territory, territory_index in player.territories.items():
            territories_state[territory_index] = 1.0

        other_player_index = 1
        for index, other_player in enumerate(self.players):
            if other_player is player:
                continue
            for territory, territory_index in other_player.territories.items():
                territories_state[territory_index +
                                  len(self.territories) * other_player_index] = 1.0
            other_player_index += 1

        armies_state = torch.zeros(
            len(self.territories) + 1, dtype=torch.float)
        for territory, territory_armies in self.armies.items():
            armies_state[self.territory_to_index[territory]
                         ] = float(territory_armies)

        armies_state[len(self.territories)] = float(armies_used)

        action_state = torch.zeros(len(Action), dtype=torch.float)
        action_state[action_to_state_index[action]] = 1.0

        # each card has a territory, a design, and a wildcard status, and you can have at most 8 cards
        card_encoding_length = len(self.territories) + 4
        infantry_offset = card_encoding_length + 1
        cavalry_offset = card_encoding_length + 2
        artillery_offset = card_encoding_length + 3
        wildcard_offset = card_encoding_length + 4
        cards_state = torch.zeros(card_encoding_length * 9, dtype=torch.float)
        for index, card in enumerate(player.hand):
            if card.wildcard:
                cards_state[wildcard_offset +
                            card_encoding_length * index] = 1.0
                continue
            cards_state[self.territory_to_index[card.territory] +
                        card_encoding_length * index] = 1.0
            match(card.design):
                case Design.INFANTRY:
                    cards_state[infantry_offset +
                                card_encoding_length * index] = 1.0
                case Design.CAVALRY:
                    cards_state[cavalry_offset +
                                card_encoding_length * index] = 1.0
                case Design.ARTILLERY:
                    cards_state[artillery_offset +
                                card_encoding_length * index] = 1.0

        selection_state = torch.zeros(
            len(self.territories), dtype=torch.float)
        for territory in selected_territories:
            selection_state[self.territory_to_index[territory]] = 1.0
        return torch.cat((territories_state, armies_state, cards_state, selection_state)).to(DEVICE)


class Risk:
    """
    A game of Risk, which handles turn order, soliciting actions from Players, 
    and applying those actions to the Board.
    """

    def __init__(self, players: list[Player], rules: Rules, board: Board):
        """
        A Risk game needs players, rules, and a board to play on.

        :params:\n
        players     --  a list of Players\n
        rules       --  a Rules\n
        board       --  a Board
        """
        self.players = [validate_is_type(player, Player) for player in players]
        self.rules = validate_is_type(rules, Rules)
        self.board = validate_is_type(board, Board)
        self.territory_indices = {territory: index for index,
                                  territory in enumerate(list(board.territories))}
        self.index_to_player = {index: player for index,
                                player in enumerate(self.players)}

    def tradein(self, player, quiet=True) -> int:
        """
        Handles the logic for players trading in cards

        :params:\n
        player      --  the player trading in the cards\n
        quiet       --  whether game updates to console are muted

        :returns:
        card_armies --  the number of armies awarded for turning in cards
        """
        matches = self.rules.get_matching_cards(player.hand)
        cards = player.use_cards(self.board, matches)

        card_armies = -1

        if cards is not None:
            card_armies = self.rules.get_armies_from_card_match(
                self.board.matches_traded)

            extra_deployments = [
                card.territory for card in cards if card.territory in player.territories]

            extra_deployment_message = '.'

            if len(extra_deployments) != 0:
                extra_armies_territory = player.choose_extra_deployment(
                    self.board, extra_deployments)

                self.board.add_armies(extra_armies_territory, 2)
                extra_deployment_message = f', plus two extra armies on {extra_armies_territory.name}'

            if not quiet:
                print(
                    f'{player.name} traded in three cards for {card_armies} armies' + extra_deployment_message)

            self.board.return_and_shuffle(*cards)
            player.remove_cards(*cards)

            # self.board.matches_traded += 1

        return card_armies

    def placement(self, player: Player, armies_awarded: int, quiet=True):
        """
        Handles players placing armies on their territories

        :params:\n
        player          --  the player placing the armies\n
        armies_awarded  --  the number of armies to place
        """
        while armies_awarded != 0:
            territory, armies_placed = player.place_armies(
                self.board, armies_awarded)
            self.board.add_armies(territory, armies_placed)
            armies_awarded -= armies_placed
            if not quiet:
                print(
                    f'{player.name} placed {armies_placed} armies on {territory.name}.')

    def play(self, quiet=True):
        """
        TODO: document this
        """

        # Initial army placement
        initial_armies = self.rules.get_initial_armies(len(self.players))
        player_order = [index for index, player in enumerate(self.players)]
        random.shuffle(player_order)
        starting_armies = {player: initial_armies for player in player_order}

        free_territories = set(self.board.territories.keys())
        last_player = None

        while len(free_territories) != 0:
            for player_index in player_order:
                player = self.index_to_player[player_index]
                claim = player.get_claim(self.board, free_territories)
                starting_armies[player_index] -= 1
                free_territories.remove(claim)
                self.board.claim(claim, player)
                player.add_territory(
                    claim, self.board.territory_to_index[claim])
                if len(free_territories) == 0:
                    last_player = player_index
                    break

        self.fix_player_order(player_order, last_player)

        while any([armies > 0 for armies in starting_armies.values()]):
            for player_index in player_order:
                player = self.index_to_player[player_index]
                if starting_armies[player_index] == 0:
                    continue
                territory, armies_placed = player.place_armies(self.board, 1)
                starting_armies[player_index] -= 1
                self.board.add_armies(territory, armies_placed)
                last_player = player_index

        self.fix_player_order(player_order, last_player)

        self.board.shuffle_deck()

        gaming = True

        rounds = 0
        dead_players = list()
        while gaming:
            if rounds > 10000:
                raise RuntimeError('Game went on too long!')

            for player_index in player_order:
                won = False
                player = self.index_to_player[player_index]
                if player in dead_players:
                    continue

                earned_card = False

                if not quiet:
                    print(f"{player.name}'s turn!")

                armies_awarded = self.rules.get_armies_from_territories_occupied(
                    player.occupied_territories())

                if not quiet:
                    print(
                        f'{player.name} gets {armies_awarded} armies from owned territories.')

                player_occupied_territories = player.territories.keys()

                for continent in self.board.continents:
                    if continent.territories.issubset(player_occupied_territories):
                        armies_awarded += continent.armies_awarded
                        if not quiet:
                            print(
                                f'{player.name} gets {continent.armies_awarded} armies from controlling {continent.name}')

                trading = True
                while trading:
                    if len(player.hand) > 2:
                        trade_armies = self.tradein(player, quiet)
                        if trade_armies == -1:
                            trading = False
                            break
                        armies_awarded += self.tradein(player, quiet)
                    else:
                        break

                self.placement(player, armies_awarded, quiet)

                attacking = True
                while attacking:
                    target, base, armies_to_attack = player.attack(self.board)
                    if target is None:
                        attacking = False
                        break
                    targeted_player = self.board.territory_owners[target]
                    armies_to_defend = targeted_player.defend(
                        self.board, target, armies_to_attack)

                    if not quiet:
                        print(
                            f'{player.name} attacks {target.name} ({self.board.armies[target]} armies) from {base.name} ({self.board.armies[base]} armies) with {armies_to_attack} armies. {targeted_player.name} defends with {armies_to_defend} armies.')

                    attacker_rolls = [random.randint(1, 6)
                                      for i in range(armies_to_attack)]
                    defender_rolls = [random.randint(1, 6)
                                      for i in range(armies_to_defend)]

                    if not quiet:
                        print(
                            f'{player.name} rolls: {attacker_rolls}; {targeted_player.name} rolls: {defender_rolls}')

                    attacker_losses, defender_losses = self.rules.resolve_attack(
                        attacker_rolls, defender_rolls)
                    self.board.add_armies(base, -attacker_losses)
                    self.board.add_armies(target, -defender_losses)

                    if not quiet:
                        print(
                            f'attacker losses: {attacker_losses}; defender losses: {defender_losses}\n{base.name} now has {self.board.armies[base]} armies; {target.name} now has {self.board.armies[target]} armies.')

                    if self.board.armies[target] == 0:
                        armies_moved = player.capture(
                            self.board, target, base, armies_to_attack - attacker_losses)
                        self.board.territory_owners[target] = player
                        player.add_territory(
                            target, self.board.territory_to_index[target])
                        targeted_player.remove_territory(target)
                        self.board.set_armies(target, armies_moved)
                        self.board.add_armies(base, -armies_moved)

                        if not earned_card:
                            earned_card = True

                        if not quiet:
                            print(
                                f'{player.name} has captured {target.name}. {target.name} now has {armies_moved} armies. {base.name} now has {self.board.armies[base]} armies.')

                        if self.board.armies[target] < 0:
                            raise RuntimeError(
                                f'Attack by {player.name} resulted in negative number of armies on {target.name}')

                        if targeted_player.is_lose():
                            dead_players.append(targeted_player)
                            dead_player_cards = targeted_player.hand
                            player.add_cards(*dead_player_cards)
                            targeted_player.remove_cards(*dead_player_cards)

                            while len(player.hand) >= 6:
                                armies_awarded = 0
                                armies_awarded += self.tradein(player, quiet)
                                self.placement(player, armies_awarded)

                            if not quiet:
                                print(
                                    f'{player.name} has eliminated {targeted_player.name}!')

                        if self.board.is_win(player):
                            gaming = False
                            won = True
                            attacking = False

                            if not quiet:
                                print(f'{player.name} has won the game!')

                if won:
                    break

                if earned_card:
                    player.add_cards(self.board.draw())

                destination, source, armies = player.fortify(self.board)
                if destination is not None:
                    self.board.add_armies(destination, armies)
                    self.board.add_armies(source, -armies)

                    if not quiet:
                        print(
                            f'{player.name} has fortified {destination.name} with {armies} armies from {source.name}.')

                    if self.board.armies[source] < 1:
                        raise RuntimeError(
                            f'{player.name} fortifying {destination.name} has left fewer than 1 army on {source.name}.')
                elif not quiet and destination is None:
                    print(f'{player.name} chose not to fortify.')

            rounds += 1

    @staticmethod
    def fix_player_order(player_order: list[Player], starting_player: Player):
        while player_order[0] != starting_player:
            player_order.append(player_order.pop(0))
