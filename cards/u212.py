from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U212(Unit): # Grim Couriers
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD], 5, 7, 3)

class U212Test(CardTestCase):
    def test_ability(self):
        card = U212()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 1))
        self.assertEqual(card.strength, U212().strength)