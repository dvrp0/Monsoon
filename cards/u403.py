from card import Card
from enums import Faction, UnitType, TriggerType, StatusEffect
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U403(Unit): # Brood Sages
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.TOAD], 2, 5, 0, TriggerType.ON_PLAY)
        self.ability_amount = 4
        self.ability_strength = 1

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position,
            Target(Target.Kind.UNIT, Target.Side.ANY, status_effects=[StatusEffect.POISONED]))

        for target in targets:
            tiles = [tile for tile in self.player.board.get_bordering_tiles(target) if self.player.board.at(tile) is None]
            self.player.random.shuffle(tiles)

            for tile in tiles[:self.ability_amount]:
                self.player.board.spawn_token_unit(self.player, tile, self.ability_strength, [UnitType.TOAD])

class U403Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 5)
        self.board.spawn_token_unit(self.local, Point(0, 3), 5).poison()
        self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 1).poison()
        self.board.spawn_token_unit(self.remote, Point(2, 2), 1).poison()
        self.board.spawn_token_unit(self.local, Point(2, 4), 1).poison()
        card = U403()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertIsNotNone(self.board.at(Point(0, 2)))
        self.assertIsNotNone(self.board.at(Point(1, 1)))
        self.assertIsNotNone(self.board.at(Point(2, 1)))
        self.assertIsNotNone(self.board.at(Point(3, 2)))
        self.assertIsNotNone(self.board.at(Point(2, 3)))
        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertIsNotNone(self.board.at(Point(3, 4)))