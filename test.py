import uuid
from enums import PlayerOrder, UnitType
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
            local_card = Unit([UnitType.ANCIENT], 0, 5, 1)
            local_card.card_id = str(uuid.uuid4())[:4]
            local_deck.append(local_card)

            remote_card = Unit([UnitType.ANCIENT], 0, 5, 1)
            remote_card.card_id = str(uuid.uuid4())[:4]
            remote_deck.append(remote_card)

        self.local = Player(local_deck, PlayerOrder.FIRST)
        self.remote = Player(remote_deck, PlayerOrder.SECOND)
        self.board = Board(self.local, self.remote)

    @SkipTest
    def test_ability(self):
        pass