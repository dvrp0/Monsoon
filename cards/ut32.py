from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT32(Unit): # Operators
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 8, 26, 1)

class UT32Test(CardTestCase):
    def test_ability(self):
        card = UT32()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, UT32().strength)