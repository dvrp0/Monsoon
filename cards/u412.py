from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U412(Unit): # Harpies of the Hunt
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.RAVEN], 3, 9, 0)

class U412Test(CardTestCase):
    def test_ability(self):
        card = U412()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))
        self.assertEqual(card.strength, U412().strength)