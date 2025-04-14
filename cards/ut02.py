from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT02(Unit): # Rapid Mousers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 4, 5, 3)

class UT02Test(CardTestCase):
    def test_ability(self):
        card = UT02()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 1))
        self.assertEqual(card.strength, UT02().strength)