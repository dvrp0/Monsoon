from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U003(Unit): # Veterans of War
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.KNIGHT], 7, 19, 1)

class U003Test(CardTestCase):
    def test_ability(self):
        card = U003()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, U003().strength)