from point import Point
from structure import Structure
from test import CardTestCase

class B001(Structure): # Fort of Ebonrock
    def __init__(self):
        super().__init__(3, 9)

class B001Test(CardTestCase):
    def test_ability(self):
        card = B001()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).card_id, "b001")
        self.assertEqual(self.board.at(Point(0, 4)).strength, 9)