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
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.DRAGON]))

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