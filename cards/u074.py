from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U074(Unit): # Lunatic Lunas
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE, UnitType.ELDER], 4, 8, 1, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_front_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY), self.player)

        if len(targets) > 0:
            self.force_attack(targets[-1])

class U074Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.remote, Point(2, 3), 3)
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 1), 3)
        u3 = self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        s1 = self.board.spawn_token_structure(self.remote, Point(1, 0), 1)
        card = U074()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(card.position, Point(1, 1))
        self.assertLessEqual(u1.strength, 0)
        self.assertLessEqual(u2.strength, 0)
        self.assertEqual(u3.strength, 1)
        self.assertEqual(s1.strength, 1)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(2, 0), 1)
        u3 = self.board.spawn_token_structure(self.remote, Point(2, 2), 1)
        card = U074()
        card.player = self.local
        card.play(Point(2, 4))

        self.assertEqual(card.position, Point(1, 4))
        self.assertLessEqual(u1.strength, 0)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(2, 0), 1)
        u3 = self.board.spawn_token_structure(self.local, Point(2, 2), 1)
        card = U074()
        card.player = self.local
        card.play(Point(2, 4))

        self.assertEqual(card.position, Point(1, 4))
        self.assertLessEqual(u1.strength, 0)