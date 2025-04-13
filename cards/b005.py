from enums import Faction
from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B005(Structure): # Temple of Time
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, 3, 7)
        self.ability_targets = 3
        self.ability_remembered = []

    def activate_ability(self, position: Point | None = None):
        tiles = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.ANY, Target.Side.FRIENDLY))

        if self.ability_remembered == []:
            self.ability_remembered = [self.player.board.at(tile).copy() for tile in tiles]
        else:
            count = 0

            for entity in self.ability_remembered:
                if self.player.board.at(entity.position) is None or self.player.board.at(entity.position) == entity:
                    self.player.board.set(entity.position, entity)
                    count += 1

                    if count >= self.ability_targets:
                        break

            self.ability_remembered = []

class B005Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 2), 10)
        self.board.spawn_token_unit(self.local, Point(1, 2), 1)
        self.board.at(Point(1, 2)).poison()
        self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        self.board.at(Point(0, 3)).vitalize()
        self.board.at(Point(0, 3)).freeze()
        self.board.spawn_token_unit(self.local, Point(1, 2), 3)
        self.board.spawn_token_unit(self.local, Point(2, 2), 3)
        self.board.spawn_token_unit(self.local, Point(2, 3), 3)
        self.board.spawn_token_unit(self.local, Point(2, 4), 3)
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
        self.board.set(Point(2, 2), None)
        self.board.set(Point(2, 3), None)
        self.board.set(Point(2, 4), None)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 5)
        card.activate_ability()

        self.assertEqual(self.board.at(Point(0, 2)).player, self.local)
        self.assertEqual(self.board.at(Point(0, 2)).strength, 10)
        self.assertEqual(self.board.at(Point(1, 2)).player, self.remote)
        self.assertEqual(self.board.at(Point(0, 3)).strength, 1)
        self.assertEqual(self.board.at(Point(0, 3)).player, self.local)
        self.assertEqual(self.board.at(Point(0, 4)), None)
        self.assertEqual(self.board.at(Point(2, 2)).player, self.local)
        self.assertEqual(self.board.at(Point(2, 2)).strength, 3)
        self.assertEqual(self.board.at(Point(2, 3)), None)
        self.assertEqual(self.board.at(Point(2, 4)), None)