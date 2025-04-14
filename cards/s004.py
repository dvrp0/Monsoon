from enums import Faction, UnitType, TriggerType
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S004(Spell): # Rain of Frogs
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, 3)
        self.ability_min_amount = 6
        self.ability_max_amount = 6

    def activate_ability(self, position: Point | None = None):
        tiles = []

        for y in range(self.player.front_line, 5):
            for x in range(4):
                if self.player.board.at(Point(x, y)) is None:
                    tiles.append(Point(x, y))

        if len(tiles) > 0:
            amount = self.player.random.randint(self.ability_min_amount, self.ability_max_amount + 1)
            for tile in tiles[:amount]:
                self.player.board.spawn_token_unit(self.player, tile, 1, [UnitType.TOAD])

class S004Test(CardTestCase):
    def test_ability(self):
        card = S004()
        card.player = self.local
        card.play()

        self.assertIsNotNone(self.board.at(Point(0, 4)))
        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertIsNotNone(self.board.at(Point(2, 4)))
        self.assertIsNotNone(self.board.at(Point(3, 4)))

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 2), 1, [UnitType.DRAGON])
        card.play()

        count = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, unit_types=[UnitType.TOAD]))
        self.assertEqual(len(count), 6)