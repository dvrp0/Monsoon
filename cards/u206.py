from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U206(Unit): # Restless Goats
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR], 2, 5, 2, TriggerType.ON_DEATH)
        self.ability_damage = 3

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.player.deal_damage(self.ability_damage)

class U206Test(CardTestCase):
    def test_ability(self):
        card = U206()
        card.player = self.local
        card.play(Point(0, 1))

        self.assertEqual(self.local.strength, 20 - card.ability_damage)
        self.assertEqual(self.remote.strength, 20 - U206().strength)

        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_structure(self.local, Point(0, 1), 1)
        self.board.spawn_token_unit(self.local, Point(1, 1), 1)
        card = U206()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.board.at(Point(0, 2)).card_id, "u206")