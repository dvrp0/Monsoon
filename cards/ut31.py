from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT31(Unit): # Agents in Charge
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.RODENT], 5, 7, 3)

class UT31Test(CardTestCase):
    def test_ability(self):
        card = UT31()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 1))
        self.assertEqual(card.strength, UT31().strength)