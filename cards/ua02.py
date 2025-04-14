from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UA02(Unit): # Eternal Ethereals
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ANCIENT], 8, 12, 3)

class UA02Test(CardTestCase):
    def test_ability(self):
        card = UA02()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 1))
        self.assertEqual(card.strength, UA02().strength)