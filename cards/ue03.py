from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from target import Context
from point import Point
from test import CardTestCase

class UE03(Unit): # Beards of Crowglyph
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ELDER], 6, 13, 0, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        tiles = [tile for tile in self.player.board.get_bordering_tiles(
            Context(self.position, source=self)
        ) if self.player.board.at(tile) is None]

        if len(tiles) > 0:
            self.player.board.spawn_token_unit(self.player, self.player.random.choice(tiles), self.strength, [UnitType.ELDER])

class UE03Test(CardTestCase):
    def test_ability(self):
        card = UE03()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(1)

        self.assertTrue(self.board.at(Point(0, 3)) is not None or self.board.at(Point(1, 4)) is not None)

        self.board.clear()
        self.board.spawn_token_structure(self.remote, Point(0, 3), 1)
        card.play(Point(0, 4))
        card.deal_damage(1)

        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertEqual(self.board.at(Point(1, 4)).strength, card.strength)