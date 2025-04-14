from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class UE31(Unit): # Scrapped Planners
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT, UnitType.ELDER], 4, 8, 1, TriggerType.AFTER_SURVIVING)
        self.ability_strength = 6

    def activate_ability(self, position: Point | None = None):
        self.strength = self.ability_strength

class UE31Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(0, 3), UE31().strength - 1)
        card = UE31()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, card.ability_strength)

        self.board.spawn_token_unit(self.remote, Point(3, 3), 1)
        card = UE31()
        card.player = self.local
        card.play(Point(3, 4))

        self.assertEqual(card.strength, card.ability_strength)