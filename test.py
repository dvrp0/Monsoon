import time, numpy as np
from enums import PlayerOrder, Faction, UnitType
from unittest import TestCase, SkipTest
from player import Player
from board import Board
from cards import *
from unit import Unit

class CardTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        random = np.random.RandomState(int(time.time()))
        local_deck, remote_deck = [], []
        unit_types = list(UnitType)
        factions = list(Faction)

        for _ in range(12):
            local_card = Unit(Faction.NEUTRAL, [UnitType(random.choice(unit_types))], random.randint(0, 9), random.randint(1, 10), random.randint(0, 3))
            local_card.card_id = f"f{str(local_card.unit_types[0].value).zfill(3)}"
            local_deck.append(local_card)

            remote_card = Unit(Faction.NEUTRAL, [UnitType(random.choice(unit_types))], random.randint(0, 9), random.randint(1, 10), random.randint(0, 3))
            remote_card.card_id = f"f{str(local_card.unit_types[0].value).zfill(3)}"
            remote_deck.append(remote_card)

        self.local = Player(Faction(random.choice(factions)), local_deck, PlayerOrder.FIRST, random)
        self.remote = Player(Faction(random.choice(factions)), remote_deck, PlayerOrder.SECOND, random)
        self.board = Board(self.local, self.remote, random)

    @SkipTest
    def test_ability(self):
        pass