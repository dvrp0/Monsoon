from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U314(Unit): # Sound Drivers
    def __init__(self):
        super().__init__([UnitType.RODENT], 3, 7, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_front_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))

        if len(targets) > 0:
            self.player.board.at(targets[0]).push(self.position)

class U314Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 5)
        card = U314()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertNotEqual(self.board.at(Point(0, 0)), None)
        self.assertEqual(self.local.front_line, 1)

        self.board.spawn_token_unit(self.local, Point(0, 1), 1)
        card = U314()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(self.board.at(Point(0, 1)).strength, 1)

        self.board.at(Point(0, 1)).destroy()
        card = U314()
        card.player = self.local
        card.play(Point(0, 3))

        self.assertNotEqual(self.board.at(Point(0, 1)), None)

        self.board.spawn_token_structure(self.local, Point(1, 3), 3)
        card = U314()
        card.player = self.local
        card.play(Point(1, 4))

        self.assertEqual(self.board.at(Point(1, 3)).strength, 3)

        self.board.spawn_token_unit(self.remote, Point(3, 3), 10)
        card = U314()
        card.player = self.local
        card.play(Point(3, 4))

        self.assertEqual(self.board.at(Point(3, 3)).strength, 10)