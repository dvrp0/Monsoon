from unit import Unit
from enums import UnitType, TriggerType
from point import Point
from test import CardTestCase

class U061(Unit): # Sparkly Kitties
    def __init__(self):
        super().__init__([UnitType.FELINE], 2, 6, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        self.confuse()
        self.gain_speed(2)

class U061Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(1, 4), 3)
        card = U061()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 3))
        self.assertEqual(card.strength, 3)

        card = U061()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 4))