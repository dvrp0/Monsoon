from card import Card
from enums import Faction
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S104(Spell): # Icicle Burst
    def __init__(self):
        super().__init__(Faction.WINTER, 2, Target(Target.Kind.UNIT, Target.Side.ENEMY))
        self.ability_damage = 12

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        target = self.player.board.at(position)

        if target.is_frozen:
            target.deal_damage(self.ability_damage, source=self)
        else:
            target.freeze()

class S104Test(CardTestCase):
    def test_ability(self):
        card = S104()
        card.player = self.local
        self.board.spawn_token_unit(self.remote, Point(0, 4), 5)
        self.board.spawn_token_unit(self.remote, Point(1, 4), card.ability_damage - 1)
        self.board.at(Point(1, 4)).freeze()
        card.play(Point(0, 4))
        card.play(Point(1, 4))

        self.assertTrue(self.board.at(Point(0, 4)).is_frozen)
        self.assertEqual(self.board.at(Point(1, 4)), None)