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
            self.player.board.spawn_token_unit(self.player.opponent, tile, self.ability_strength, [UnitType.RAVEN])

class U406Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 7)
        card = U406()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertEqual(self.board.at(Point(1, 4)).player, self.remote)
        self.assertEqual(self.board.at(Point(1, 4)).strength, card.ability_strength)

        self.board.clear()
        card = U406()
        card.player = self.local
        card.play(Point(2, 4))
        card.destroy()

        u1 = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ANY))[0]
        self.assertTrue(u1 in [Point(2, 2), Point(1, 3), Point(3, 3), Point(2, 4)])
        self.assertEqual(self.board.at(u1).player, self.remote)
        self.assertEqual(self.board.at(u1).strength, card.ability_strength)