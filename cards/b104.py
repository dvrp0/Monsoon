from card import Card
from enums import Faction
from point import Point
from structure import Structure
from target import Context, Target
from test import CardTestCase

class B104(Structure): # Glacier Palace
    def __init__(self):
        super().__init__(Faction.WINTER, 4, 8)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        if len(targets) > 0:
            targets.sort(key=lambda t: (t.y, self.player.random.random()), reverse=True)
            self.player.board.at(targets[0]).freeze()

class B104Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 4), 1)
        card = B104()
        card.player = self.local
        card.play(Point(0, 3))
        card.activate_ability()

        self.assertFalse(u1.is_frozen)
        self.assertEqual(u2.is_frozen + u3.is_frozen + u4.is_frozen, 1)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(3, 3), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(3, 0), 1)
        card = B104()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertTrue(u1.is_frozen)
        self.assertFalse(u2.is_confused)
        self.assertFalse(u3.is_confused)