from enums import UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U206(Unit): # Restless Goats
    def __init__(self):
        super().__init__([UnitType.SATYR], 2, 5, 2, TriggerType.ON_DEATH)

    def activate_ability(self, position: Point | None = None):
        self.player.deal_damage(2)

class U206Test(CardTestCase):
    def test_ability(self):
        card = U206()
        card.player = self.local
        card.play(Point(0, 1))

        self.assertEqual(self.local.strength, 18)
        self.assertEqual(self.remote.strength, 15)

        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_structure(self.local, Point(0, 1), 1)
        self.board.spawn_token_unit(self.local, Point(1, 1), 1)
        card = U206()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.board.at(Point(0, 2)).card_id, "u206")