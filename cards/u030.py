from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U030(Unit): # Cabin Girls
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PIRATE], 4, 11, 0)

class U030Test(CardTestCase):
    def test_ability(self):
        card = U030()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, U030().strength)