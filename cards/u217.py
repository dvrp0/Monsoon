from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U217(Unit): # Gathering Troupe
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR, UnitType.ANCIENT], 4, 5, 1, TriggerType.BEFORE_MOVING)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None):
        tiles = [tile for tile in self.player.board.get_row_tiles(Point(0, 4)) if self.player.board.at(tile) is None]

        if len(tiles) > 0:
            self.player.board.spawn_token_unit(self.player, self.player.random.choice(tiles), self.ability_strength, [UnitType.SATYR])

class U217Test(CardTestCase):
    def test_ability(self):
        card = U217()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(self.board.at(Point(1, 4)) is not None or self.board.at(Point(2, 4)) is not None or self.board.at(Point(3, 4)) is not None)

        self.board.clear()
        s1 = self.board.spawn_token_structure(self.remote, Point(0, 4), 3)
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 4), 2)
        self.board.spawn_token_unit(self.local, Point(3, 4), 1)
        card = U217()
        card.player = self.local
        card.play(Point(2, 4))

        self.assertEqual(self.board.at(Point(0, 4)).card_id, s1.card_id)
        self.assertEqual(card.position, Point(1, 4))
        self.assertIsNone(self.board.at(Point(2, 4)))