from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U031(Unit): # Westwind Sailors
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PIRATE], 3, 7, 1)

class U031Test(CardTestCase):
    def test_ability(self):
        card = U031()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, U031().strength)