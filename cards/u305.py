import random
from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U305(Unit): # Linked Golems
    def __init__(self):
        super().__init__([UnitType.CONSUTRUCT], 3, 3, 1, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.CONSUTRUCT]))

        if len(targets) > 0:
            self.player.board.at(random.choice(targets)).heal(4)
            self.heal(4)

class U305Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 5, [UnitType.CONSUTRUCT])
        self.board.spawn_token_unit(self.local, Point(2, 4), 5, [UnitType.CONSUTRUCT])

        card = U305()
        card.player = self.local
        card.play(Point(1, 4))

        self.assertTrue(self.board.at(Point(0, 4)).strength == 9 or self.board.at(Point(2, 4)).strength == 9)
        self.assertEqual(card.strength, 7)