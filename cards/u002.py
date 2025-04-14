from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U002(Unit): # Heroic Soldiers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.KNIGHT], 5, 12, 1)

class U002Test(CardTestCase):
    def test_ability(self):
        card = U002()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))
        self.assertEqual(card.strength, U002().strength)