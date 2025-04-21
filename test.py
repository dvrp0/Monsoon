import time, numpy as np
from enums import PlayerOrder, Faction, UnitType
from unittest import TestCase, SkipTest
from player import Player
from point import Point
from collections.abc import Callable
from board import Board
from cards import *
from unit import Unit

class CardTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        random = np.random.RandomState(int(time.time()))
        local_deck, remote_deck = [], []
        unit_types = list(UnitType)
        factions = list(Faction)[1:] # Excluding Neutral

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

    def hook(self, instance: object, method_name: str, before: Callable | None = None, after: Callable | None = None):
        method = getattr(instance, method_name)
        orig_name = f"_orig_{method_name}"
        setattr(instance, orig_name, method)

        def wrapper(self, *args, **kwargs):
            if before:
                before(self, *args, **kwargs)

            getattr(self, orig_name)(*args, **kwargs)

            if after:
                after(self, *args, **kwargs)

        setattr(instance, method_name, wrapper.__get__(instance))

class BaseTestCase(CardTestCase):
    def test_ability(self):
        from target import Target
        from cards.u401 import U401
        from cards.u405 import U405
        from cards.ue42 import UE42
        from cards.s021 import S021

        # Trigger resolving order
        positions = []

        for y in range(1, 5):
            for x in range(0, 4):
                u401 = U401()
                u401.player = self.local

                self.hook(u401, "activate_ability", lambda s: positions.append(s.position))
                self.board.set(Point(x, y), u401)

        self.board.at(Point(0, 1)).deal_damage(1)

        self.assertEqual(len(self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ANY))), 0)
        self.assertEqual(positions, [
            Point(0, 1),
            Point(0, 2),
            Point(0, 3),
            Point(0, 4),
            Point(1, 4),
            Point(2, 4),
            Point(3, 4),
            Point(3, 3),
            Point(3, 2),
            Point(2, 2),
            Point(2, 1),
            Point(3, 1),
            Point(2, 3),
            Point(1, 3),
            Point(1, 2),
            Point(1, 1),
        ])

        positions = []

        u401 = U401()
        u401.player = self.local
        self.hook(u401, "activate_ability", lambda s: positions.append(s.position))
        self.board.set(Point(1, 3), u401)
        ue42 = UE42()
        ue42.player = self.local
        self.hook(ue42, "activate_ability", lambda s: positions.append(s.position))
        self.board.set(Point(3, 3), ue42)
        u405 = U405()
        u405.player = self.local
        u405.play(Point(2, 3))

        self.assertEqual(positions, [
            Point(3, 3),
            Point(1, 3)
        ])

        positions = []

        u401 = U401()
        u401.player = self.remote
        self.hook(u401, "activate_ability", lambda s: positions.append(s.position))
        self.board.set(Point(1, 3), u401)
        ue42 = UE42()
        ue42.player = self.remote
        self.hook(ue42, "activate_ability", lambda s: positions.append(s.position))
        self.board.set(Point(3, 3), ue42)
        u405 = U405()
        u405.player = self.local
        u405.play(Point(2, 3))

        self.assertEqual(positions, [
            Point(3, 3),
            Point(1, 3)
        ])