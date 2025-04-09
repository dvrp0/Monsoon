from enums import Faction, UnitType
from point import Point
from unit import Unit
from test import CardTestCase

class U001(Unit): # Gifted Recruits
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.KNIGHT], 2, 5, 1)

class U001Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(1, 4), 1)
        card = U001()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(1, 4)).card_id, "u001")
        self.assertEqual(self.board.at(Point(1, 4)).strength, 4)