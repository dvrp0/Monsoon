from enums import Faction, UnitType
from point import Point
from unit import Unit
from test import CardTestCase

class U025(Unit): # Lawless Herd
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.SATYR], 2, 6, 0)

class U025Test(CardTestCase):
    def test_ability(self):
        card = U025()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).card_id, "u025")
        self.assertEqual(self.board.at(Point(0, 4)).strength, 6)