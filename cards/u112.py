from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U112(Unit): # Calming Spirits
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE], 7, 26, 0)

class U112Test(CardTestCase):
    def test_ability(self):
        card = U112()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, U112().strength)