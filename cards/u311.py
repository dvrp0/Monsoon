from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U311(Unit): # Delegators
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 6, 16, 1)

class U311Test(CardTestCase):
    def test_ability(self):
        card = U311()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, U311().strength)