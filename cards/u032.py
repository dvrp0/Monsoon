from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U032(Unit): # Bluesail Raiders
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PIRATE], 5, 8, 2)

class U032Test(CardTestCase):
    def test_ability(self):
        card = U032()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, U032().strength)