import random
from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B203(Structure): # Temple of Focus
    def __init__(self):
        super().__init__(3, 7)

    def activate_ability(self, position: Point | None = None):
        for tile in self.player.board.get_front_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY)):
            target = self.player.board.at(tile)

            if target.is_confused:
                target.deconfuse()
            target.command()

class B203Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 5)
        self.board.at(Point(0, 3)).confuse()
        self.board.spawn_token_unit(self.local, Point(0, 0), 7)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 3), 6)
        self.board.spawn_token_unit(self.remote, Point(1, 0), 6)
        card = B203()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(self.board.at(Point(0, 3)), None)
        self.assertEqual(self.board.at(Point(0, 2)).strength, 4)
        self.assertFalse(self.board.at(Point(0, 2)).is_confused)
        self.assertEqual(self.board.at(Point(0, 0)), None)
        self.assertEqual(self.remote.strength, 13)