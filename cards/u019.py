from enums import Faction, UnitType
from unit import Unit
from point import Point
from test import CardTestCase

class U019(Unit): # Headless Hotheads
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.CONSUTRUCT], 5, 7, 2, fixedly_forward=True)

class U019Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 3), 1)
        u3 = self.board.spawn_token_structure(self.remote, Point(0, 2), 1)
        card = U019()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(card.strength, U019().strength - 1)
        self.assertEqual(u1.strength, 1)
        self.assertEqual(u2.strength, 1)
        self.assertLessEqual(u3.strength, 0)