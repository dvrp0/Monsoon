from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class UT42(Unit): # Untamed Cultists
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.RAVEN], 7, 26, 0)

class UT42Test(CardTestCase):
    def test_ability(self):
        card = UT42()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, UT42().strength)