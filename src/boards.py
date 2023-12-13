from riskLogic import Player, Board
from validators import *
from classicGame import classic_territories, classic_continents


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
