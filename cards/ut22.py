from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT22(Unit): # Mindless Horde
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR], 3, 8, 0)

class UT22Test(CardTestCase):
    def test_ability(self):
        card = UT22()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, UT22().strength)