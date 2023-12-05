import random
from enums import PlayerOrder, Faction, UnitType
from unittest import TestCase, SkipTest
from player import Player
from board import Board
from cards import *
from unit import Unit

class CardTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        local_deck, remote_deck = [], []

        for _ in range(12):
            local_card = Unit([random.choice(list(UnitType))], random.randint(0, 9), random.randint(1, 10), random.randint(0, 3))
            local_card.card_id = f"f{str(local_card.unit_types[0].value).zfill(3)}"
            local_deck.append(local_card)

            remote_card = Unit([random.choice(list(UnitType))], random.randint(0, 9), random.randint(1, 10), random.randint(0, 3))
            remote_card.card_id = f"f{str(local_card.unit_types[0].value).zfill(3)}"
            remote_deck.append(remote_card)

        self.local = Player(random.choice(list(Faction)), local_deck, PlayerOrder.FIRST)
        self.remote = Player(random.choice(list(Faction)), remote_deck, PlayerOrder.SECOND)
        self.board = Board(self.local, self.remote)

    @SkipTest
    def test_ability(self):
        pass