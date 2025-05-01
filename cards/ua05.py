from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from target import Context
from point import Point
from test import CardTestCase

class UA05(Unit): # Bounded Daemons
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ANCIENT], 4, 9, 0, TriggerType.BEFORE_MOVING)
        self.ability_strength = 4

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        tiles = [tile for tile in self.player.board.get_side_tiles(
            Context(self.position, source=self)
        ) if self.player.board.at(tile) is None]

        for tile in tiles:
            self.player.board.spawn_token_unit(self.player, tile, self.ability_strength, [UnitType.ANCIENT])

class UA05Test(CardTestCase):
    def test_ability(self):
        card = UA05()
        card.player = self.local
        card.play(Point(0, 4))
        card.command()

        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertEqual(self.board.at(Point(1, 4)).strength, card.ability_strength)
        self.assertEqual(self.board.at(Point(1, 4)).unit_types, [UnitType.ANCIENT])

        self.board.clear()
        card = UA05()
        card.player = self.local
        card.play(Point(2, 4))
        card.command()

        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertIsNotNone(self.board.at(Point(3, 4)))