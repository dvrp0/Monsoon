from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U216(Unit): # Reckless Rushers
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD], 4, 6, 3, TriggerType.BEFORE_ATTACKING)
        self.ability_damage = 2

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.player.deal_damage(self.ability_damage)

class U216Test(CardTestCase):
    def test_ability(self):
        card = U216()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.local.strength, 20 - card.ability_damage)
        self.assertEqual(self.remote.strength, 20 - U216().strength)

        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        card = U216()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.local.strength, 20 - card.ability_damage * 4)
        self.assertEqual(self.remote.strength, 20 - U216().strength * 2 + 2)