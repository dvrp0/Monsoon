import random
from enums import UnitType
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S012(Spell): # Summon Militia
    def __init__(self):
        super().__init__(1)

    def activate_ability(self, position: Point | None = None):
        tiles = []

        for y in range(self.player.front_line, 5):
            for x in range(4):
                if self.player.board.at(Point(x, y)) is None:
                    tiles.append(Point(x, y))

        if len(tiles) > 0:
            self.player.board.spawn_token_unit(self.player, random.choice(tiles), 5, [UnitType.KNIGHT])

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