from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT01(Unit): # Seasick Bouncers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PIRATE], 6, 20, 0)

class UT01Test(CardTestCase):
    def test_ability(self):
        card = UT01()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, UT01().strength)