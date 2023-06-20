import random
from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U306(Unit): # Destuctobots
    def __init__(self):
        super().__init__([UnitType.CONSUTRUCT], 2, 6, 1, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.FRIENDLY), self.position)

        if len(targets) > 0:
            self.player.board.at(random.choice(targets)).deal_damage(1)

class U306Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(3, 1), 5)
        self.board.spawn_token_structure(self.local, Point(0, 4), 5)
        self.board.spawn_token_unit(self.local, Point(1, 4), 5)
        card = U306()
        card.player = self.local
        card.play(Point(2, 4))

        self.assertEqual(self.board.at(Point(3, 1)).strength, 5)
        self.assertTrue(self.board.at(Point(0, 4)).strength == 4 or self.board.at(Point(1, 4)).strength == 4)