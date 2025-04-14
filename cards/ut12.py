from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT12(Unit): # Sleetstompers
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.VIKING], 8, 14, 2)

class UT12Test(CardTestCase):
    def test_ability(self):
        card = UT12()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, UT12().strength)