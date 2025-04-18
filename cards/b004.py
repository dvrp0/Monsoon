from enums import Faction
from point import Point
from target import Target
from structure import Structure
from test import CardTestCase

class B004(Structure): # Powder Tower
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 5, 8)
        self.ability_damage = 4

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY), include_base=True)

        for target in targets:
            self.player.board.at(target).deal_damage(self.ability_damage)

        self.destroy()

class B004Test(CardTestCase):
    def test_ability(self):
        card = B004()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 3), card.ability_damage)
        u2 = self.board.spawn_token_unit(self.remote, Point(2, 2), card.ability_damage + 2)
        u3 = self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        u4 = self.board.spawn_token_unit(self.local, Point(2, 1), 1)
        s1 = self.board.spawn_token_structure(self.remote, Point(3, 4), card.ability_damage)
        s2 = self.board.spawn_token_structure(self.remote, Point(3, 3), card.ability_damage + 1)
        s3 = self.board.spawn_token_structure(self.local, Point(3, 2), 1)
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertLessEqual(u1.strength, 0)
        self.assertEqual(u2.strength, 2)
        self.assertLessEqual(u3.strength, 0)
        self.assertEqual(u4.strength, 1)
        self.assertLessEqual(s1.strength, 0)
        self.assertEqual(s2.strength, 1)
        self.assertEqual(s3.strength, 1)
        self.assertEqual(self.remote.strength, 20 - card.ability_damage)
        self.assertIsNone(self.board.at(Point(0, 4)))