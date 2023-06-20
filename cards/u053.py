import random
from enums import UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U053(Unit): # Wild Saberpaws
    def __init__(self):
        super().__init__([UnitType.FELINE], 2, 5, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        if len(self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.ANY, Target.Side.ENEMY))) == 0:
            self.gain_speed(2)
        elif len(self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.ANY, Target.Side.ENEMY))) == 0:
            self.gain_speed(1)

class U053Test(CardTestCase):
    def test_ability(self):
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 3), 5)
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))