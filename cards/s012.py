from card import Card
from enums import Faction, UnitType
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S012(Spell): # Summon Militia
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 1)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        tiles = self.player.get_within_front_line()
        empty = [tile for tile in tiles if self.player.board.at(tile) is None]

        if len(empty) > 0:
            self.player.board.spawn_token_unit(self.player, self.player.random.choice(empty), self.ability_strength, [UnitType.KNIGHT])

class S012Test(CardTestCase):
    def test_ability(self):
        card = S012()
        card.player = self.local
        card.play()

        target = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.KNIGHT]))[0]
        self.assertTrue(self.board.at(target).position.y >= self.local.front_line)

        self.board.clear()
        self.local.front_line = 2
        card.play()

        target = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.KNIGHT]))[0]
        self.assertTrue(self.board.at(target).position.y >= self.local.front_line)

        self.board.clear()
        self.board.spawn_token_structure(self.remote, Point(0, 4), 5)
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        self.board.spawn_token_unit(self.remote, Point(2, 4), 5)
        self.board.spawn_token_structure(self.remote, Point(3, 4), 5)

        target = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.KNIGHT]))
        self.assertEqual(target, [])