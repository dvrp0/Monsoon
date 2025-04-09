from enums import Faction
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S105(Spell): # Blessed with Brawn
    def __init__(self):
        super().__init__(Faction.WINTER, 5, Target(Target.Kind.ANY, Target.Side.FRIENDLY, strength_limit=8))

    def activate_ability(self, position: Point | None = None):
        self.player.board.at(position).heal(15)

class S105Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.local, Point(0, 4), 8)
        card = S105()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 23)

        self.board.spawn_token_unit(self.local, Point(1, 4), 9)
        card.play(Point(1, 4))

        self.assertEqual(self.board.at(Point(1, 4)).strength, 9)