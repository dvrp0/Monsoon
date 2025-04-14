from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U020(Unit): # Warfront Runners
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.KNIGHT], 4, 7, 2)

class U020Test(CardTestCase):
    def test_ability(self):
        card = U020()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, U020().strength)