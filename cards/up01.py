from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UP01(Unit): # Eager Pursuers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PRIMAL], 4, 9, 1)

class UP01Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 3)
        card = UP01()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).card_id, "up01")
        self.assertEqual(self.board.at(Point(0, 3)).strength, UP01().strength - 3)