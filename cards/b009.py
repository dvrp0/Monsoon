from enums import Faction
from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B009(Structure): # Chateau de Cardboard
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3, 7)
        self.ability_amount = 2

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ENEMY))

        if len(targets) > 0:
            targets.sort(key=lambda t: (t.y, self.player.random.random()), reverse=True)
            targets = targets[:self.ability_amount]
            self.player.random.shuffle(targets)

            for target in targets:
                self.player.board.at(target).confuse()

class B009Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 4), 1)
        card = B009()
        card.player = self.local
        card.play(Point(0, 3))
        card.activate_ability()

        self.assertFalse(u1.is_confused)
        self.assertEqual(u2.is_confused + u3.is_confused + u4.is_confused, 2)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(3, 3), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(3, 0), 1)
        card = B009()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertTrue(u1.is_confused and u2.is_confused)
        self.assertFalse(u3.is_confused)