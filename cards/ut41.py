from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT41(Unit): # Limelimbs
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.TOAD], 3, 5, 2)

class UT41Test(CardTestCase):
    def test_ability(self):
        card = UT41()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, UT41().strength)