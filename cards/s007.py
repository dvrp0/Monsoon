from card import Card
from enums import Faction
from point import Point
from spell import Spell
from target import Context, Target
from test import CardTestCase

class S007(Spell): # Potion of Growth
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))
        self.ability_strength = 7

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        target = self.player.board.at(position)
        target.heal(self.ability_strength)
        target.vitalize()

class S007Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 7)
        card = S007()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 7 + card.ability_strength)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 4), 7)
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 7)