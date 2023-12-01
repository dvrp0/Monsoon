from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S104(Spell): # Icicle Burst
    def __init__(self):
        super().__init__(1, Target(Target.Kind.UNIT, Target.Side.ENEMY))

    def activate_ability(self, position: Point | None = None):
        target = self.player.board.at(position)

        if target.is_frozen:
            target.deal_damage(12)
        else:
            target.freeze()

class S104Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 4), 5)
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        self.board.at(Point(1, 4)).freeze()
        card = S104()
        card.player = self.local
        card.play(Point(0, 4))
        card.play(Point(1, 4))

        self.assertTrue(self.board.at(Point(0, 4)).is_frozen)
        self.assertEqual(self.board.at(Point(1, 4)), None)