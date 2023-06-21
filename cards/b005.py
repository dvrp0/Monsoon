from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B005(Structure): # Temple of Time
    def __init__(self):
        super().__init__(3, 7)
        self.remembered = []

    def activate_ability(self, position: Point | None = None):
        tiles = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.ANY, Target.Side.FRIENDLY))

        if self.remembered == []:
            self.remembered = [self.player.board.at(tile).copy() for tile in tiles]
        else:
            for entity in self.remembered:
                if self.player.board.at(entity.position) is None or self.player.board.at(entity.position) == entity:
                    self.player.board.set(entity.position, entity)

            self.remembered = []

class B005Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 2), 10)
        self.board.spawn_token_unit(self.local, Point(1, 2), 1)
        self.board.at(Point(1, 2)).poison()
        self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        self.board.at(Point(0, 3)).vitalize()
        self.board.at(Point(0, 3)).freeze()
        self.board.spawn_token_unit(self.local, Point(1, 2), 3)
        self.board.spawn_token_unit(self.remote, Point(0, 4), 5)
        card = B005()
        card.player = self.local
        card.play(Point(1, 3))
        card.activate_ability()

        self.board.at(Point(0, 2)).deal_damage(7)
        self.board.at(Point(0, 2)).poison()
        self.board.set(Point(0, 3), None)
        self.board.set(Point(0, 4), None)
        self.board.set(Point(1, 2), None)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 5)
        card.activate_ability()

        self.assertEqual(self.board.at(Point(0, 2)).player, self.local)
        self.assertEqual(self.board.at(Point(0, 2)).strength, 10)
        self.assertEqual(self.board.at(Point(1, 2)).player, self.remote)
        self.assertEqual(self.board.at(Point(0, 3)).strength, 1)
        self.assertEqual(self.board.at(Point(0, 3)).player, self.local)
        self.assertEqual(self.board.at(Point(0, 4)), None)