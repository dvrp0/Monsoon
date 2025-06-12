from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U026(Unit): # Boomstick Officers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.RODENT], 3, 6, 1, TriggerType.ON_PLAY)
        self.ability_damage = 6

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_behind_tiles(Context(self.position, source=self), Target(Target.Kind.ANY, Target.Side.ENEMY))

        if len(targets) > 0:
            self.player.board.at(targets[0]).deal_damage(self.ability_damage, source=self)

class U026Test(CardTestCase):
    def test_ability(self):
        card = U026()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(0, 4), 1)
        s1 = self.board.spawn_token_structure(self.remote, Point(0, 3), 1)
        card.play(Point(0, 1))

        self.assertLessEqual(u1.strength, 0)
        self.assertEqual(u2.strength, 1)
        self.assertEqual(s1.strength, 1)