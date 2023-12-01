import random
from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U320(Unit): # Original Blueprints
    def __init__(self):
        super().__init__([UnitType.CONSUTRUCT, UnitType.ANCIENT], 4, 7, 1, TriggerType.BEFORE_MOVING)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))

        if len(targets) > 0:
            self.player.board.at(random.choice(targets)).heal(4)

class U320Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        self.board.spawn_token_unit(self.local, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        card = U320()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertTrue(self.board.at(Point(0, 4)).strength == 5 or self.board.at(Point(1, 4)).strength == 5)
        self.assertEqual(self.board.at(Point(2, 4)).strength, 1)