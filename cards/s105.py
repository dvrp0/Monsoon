from card import Card
from enums import Faction
from point import Point
from spell import Spell
from target import Context, Target
from test import CardTestCase

class S105(Spell): # Blessed with Brawn
    def __init__(self):
        super().__init__(Faction.WINTER, 5, Target(Target.Kind.ANY, Target.Side.FRIENDLY, strength_limit=10))
        self.ability_strength = 15

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.player.board.at(position).heal(self.ability_strength)

class S105Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.local, Point(0, 4), 8)
        card = S105()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 8 + card.ability_strength)

        limit = card.required_targets.strength_limit + 1
        self.board.spawn_token_unit(self.local, Point(1, 4), limit)
        card.play(Point(1, 4))

        self.assertEqual(self.board.at(Point(1, 4)).strength, limit)