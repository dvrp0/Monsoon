from enums import UnitType
from point import Point
from unit import Unit
from test import CardTestCase

class U069(Unit): # Ultrasonics
    def __init__(self):
        super().__init__([UnitType.UNDEAD], 5, 6, 4)

class U069Test(CardTestCase):
    def test_ability(self):
        card = U069()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 0)).card_id, "u069")

        card = U069()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.remote.strength, 14)

        self.board.spawn_token_structure(self.remote, Point(1, 3), 1)
        self.board.spawn_token_structure(self.remote, Point(1, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(3, 2), 1)
        card = U069()
        card.player = self.local
        card.play(Point(0, 3))

        self.assertEqual(self.board.at(Point(1, 3)), None)
        self.assertEqual(self.board.at(Point(1, 2)), None)
        self.assertEqual(self.board.at(Point(2, 2)), None)
        self.assertEqual(self.board.at(Point(2, 3)).strength, 1)
        self.assertEqual(self.board.at(Point(3, 2)).card_id, "u069")

        self.board.spawn_token_unit(self.remote, Point(1, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 0), 1)
        card = U069()
        card.player = self.local
        card.play(Point(2, 1))

        self.assertEqual(self.board.at(Point(1, 0)).strength, 1)
        self.assertEqual(self.board.at(Point(2, 0)), None)
        self.assertEqual(self.remote.strength, 9)