from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S007(Spell): # Potion of Growth
    def __init__(self):
        super().__init__(3, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))

    def activate_ability(self, position: Point | None = None):
        target = self.player.board.at(position)
        target.heal(6)
        target.vitalize()

class S007Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 7)
        card = S007()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 13)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 4), 7)
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 7)