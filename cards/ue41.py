from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UE41(Unit): # Faithless Prophets
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.RAVEN, UnitType.ELDER], 3, 10, 1, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None):
        self.convert()

class UE41Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(1, 0), 1)
        card = UE41()
        card.player = self.local
        card.play(Point(1, 1))
        card.poison()
        self.board.to_next_turn()

        self.assertEqual(self.remote.strength, 20 - (UE41().strength - 2))

        self.board.to_next_turn()
        self.board.spawn_token_unit(self.remote, Point(1, 2), 1)
        self.board.spawn_token_structure(self.remote, Point(3, 2), 1)
        card = UE41()
        card.player = self.local
        card.play(Point(2, 2))

        self.assertEqual(card.position, Point(1, 2))