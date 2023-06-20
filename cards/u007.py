import random
from enums import UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U007(Unit): # Green Prototypes
    def __init__(self):
        super().__init__([UnitType.CONSUTRUCT], 1, 5, 1, TriggerType.ON_DEATH)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY))

        if len(targets) > 0:
            target = self.player.board.at(random.choice(targets))
            target.heal(5)
            target.vitalize()

class U007Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 6)

        card = U007()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, 6)
        self.assertTrue(self.board.at(Point(0, 3)).is_vitalized)