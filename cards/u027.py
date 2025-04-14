from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U027(Unit): # Salty Outcasts
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.TOAD], 7, 12, 2)

class U027Test(CardTestCase):
    def test_ability(self):
        card = U027()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, U027().strength)