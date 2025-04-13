from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U406(Unit): # Dubious Hags
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.RAVEN], 2, 6, 1, TriggerType.ON_DEATH)
        self.ability_strength = 1

    def activate_ability(self, position: Point | None = None):
        tiles = [tile for tile in self.player.board.get_bordering_tiles(self.position) if self.player.board.at(tile) is None]

        if len(tiles) > 0:
            tile = self.player.random.choice(tiles)
            self.player.board.spawn_token_unit(self.player, tile, self.ability_strength, [UnitType.RAVEN])

class U406Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 7)
        card = U406()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertIsNotNone(self.board.at(Point(1, 4)))

        self.board.clear()
        card = U406()
        card.player = self.local
        card.play(Point(2, 4))
        card.destroy()

        self.assertTrue(self.board.at(Point(2, 2)) is not None or self.board.at(Point(1, 3)) is not None or \
            self.board.at(Point(3, 3)) is not None or self.board.at(Point(2, 4)) is not None)