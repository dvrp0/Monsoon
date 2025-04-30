from card import Card
from enums import Faction
from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B002(Structure): # Trueshot Post
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 5, 8)
        self.ability_damage = 8

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY))

        if len(targets) > 0:
            targets.sort(key=lambda t: (t.y, self.player.random.random()), reverse=True)
            self.player.board.at(targets[0]).deal_damage(self.ability_damage, source=self)

class B002Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        strength = 20
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 4), strength)
        u3 = self.board.spawn_token_unit(self.remote, Point(2, 4), strength)
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 4), strength)
        card = B002()
        card.player = self.local
        card.play(Point(0, 3))
        card.activate_ability()

        self.assertEqual(u1.strength, 1)
        self.assertEqual(u2.strength + u3.strength + u4.strength, strength * 3 - card.ability_damage)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(2, 4), strength)
        u2 = self.board.spawn_token_unit(self.remote, Point(3, 3), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(3, 0), 1)
        card = B002()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(u1.strength, strength - card.ability_damage)
        self.assertEqual(u2.strength + u3.strength, 2)