from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UD07(Unit): # Flameless Lizards
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.DRAGON], 5, 15, 0)

class UD07Test(CardTestCase):
    def test_ability(self):
        card = UD07()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, UD07().strength)