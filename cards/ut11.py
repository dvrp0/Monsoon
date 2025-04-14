from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT11(Unit): # Iced Droplings
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE], 6, 16, 1)

class UT11Test(CardTestCase):
    def test_ability(self):
        card = UT11()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, UT11().strength)