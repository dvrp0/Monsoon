from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT21(Unit): # Obliterators
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD], 6, 10, 2)

class UT21Test(CardTestCase):
    def test_ability(self):
        card = UT21()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, UT21().strength)