from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U216(Unit): # Reckless Rushers
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD], 3, 5, 3, TriggerType.BEFORE_ATTACKING)

    def activate_ability(self, position: Point | None = None):
        self.player.deal_damage(1)

class U216Test(CardTestCase):
    def test_ability(self):
        card = U216()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.local.strength, 19)
        self.assertEqual(self.remote.strength, 15)

        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        card = U216()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.local.strength, 16)
        self.assertEqual(self.remote.strength, 12)