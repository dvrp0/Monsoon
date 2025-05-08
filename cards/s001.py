from card import Card
from enums import Faction
from point import Point
from target import Target
from spell import Spell
from test import CardTestCase

class S001(Spell): # Execution
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 4, Target(Target.Kind.UNIT, Target.Side.ENEMY))
        self.ability_damage = 9

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.player.board.at(position).deal_damage(self.ability_damage, source=self)

class S001Test(CardTestCase):
    def test_ability(self):
        card = S001()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 2), card.ability_damage)
        card.play(u1.position)

        self.assertLessEqual(u1.strength, 0)