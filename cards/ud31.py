import random
from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UD31(Unit): # Greengale Surpents
    def __init__(self):
        super().__init__([UnitType.DRAGON], 3, 3, 2, TriggerType.BEFORE_ATTACKING)

    def activate_ability(self, position: Point | None = None):
        if not position.is_valid or not isinstance(self.player.board.at(position), Unit): # 기지 또는 건물이라면 발동 안 함
            return

        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.DRAGON]), self.player)

        if len(targets) > 0:
            self.player.board.at(random.choice(targets)).heal(3)

        self.heal(3)

class UD31Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 1, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 3), 1)
        card = UD31()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, 7)
        self.assertEqual(self.board.at(Point(0, 3)).strength, 7)

        self.board.clear()
        card = UD31()
        card.player = self.local
        card.play(Point(0, 1))

        self.assertEqual(self.board.at(Point(0, 0)), None)
        self.assertEqual(self.remote.strength, 17)

        self.board.spawn_token_structure(self.remote, Point(1, 2), 10)
        card = UD31()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.board.at(Point(1, 2)).strength, 7)