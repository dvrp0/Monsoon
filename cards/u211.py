from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U211(Unit): # Doppelbocks
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR], 2, 3, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        tiles = self.player.board.get_front_tiles(self.position)

        if len(tiles) > 0 and self.player.board.at(tiles[0]) is None:
            self.player.board.spawn_token_unit(self.player, tiles[0], 3, [UnitType.SATYR])

class U211Test(CardTestCase):
    def test_ability(self):
        card = U211()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, 3)
        self.assertEqual(self.board.at(Point(0, 3)).unit_types, [UnitType.SATYR])
        self.assertEqual(self.local.front_line, 3)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 3), 6)
        card = U211()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, 6)