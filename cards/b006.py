from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B006(Structure): # Temple of Life
    def __init__(self):
        super().__init__(3, 6)

    def activate_ability(self, position: Point | None = None):
        targets = [target for target in self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY))
                   if not self.player.board.at(target).is_vitalized]
        self.player.random.shuffle(targets)

        for target in targets[:3]:
            self.player.board.at(target).vitalize()

        tiles = []
        front = self.player.board.get_front_tiles(self.position)
        behind = self.player.board.get_behind_tiles(self.position)

        if len(front) > 0 and self.player.board.at(front[0]) is None and front[0].y >= self.player.front_line:
            tiles.append(front[0])

        if len(behind) > 0 and self.player.board.at(behind[0]) is None:
            tiles.append(behind[0])

        copy = self.copy()
        copy.strength = 1
        copy.play(self.player.random.choice(tiles))

class B006Test(CardTestCase):
    def test_ability(self):
        self.local.front_line = 3
        self.board.spawn_token_unit(self.local, Point(3, 4), 1)
        self.board.spawn_token_unit(self.local, Point(3, 3), 1)
        self.board.spawn_token_unit(self.local, Point(3, 2), 1)
        card = B006()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(self.board.at(Point(0, 3)).card_id, "b006")
        self.assertEqual(self.board.at(Point(0, 3)).strength, 1)
        self.assertTrue(self.board.at(Point(3, 4)).is_vitalized)
        self.assertTrue(self.board.at(Point(3, 3)).is_vitalized)
        self.assertTrue(self.board.at(Point(3, 2)).is_vitalized)

        self.board.clear()
        self.local.front_line = 3
        card = B006()
        card.player = self.local
        card.play(Point(0, 3))
        card.activate_ability()

        self.assertEqual(self.board.at(Point(0, 4)).card_id, "b006")
        self.assertEqual(self.board.at(Point(0, 4)).strength, 1)

        self.board.clear()
        self.local.front_line = 2
        card = B006()
        card.player = self.local
        card.play(Point(0, 3))
        card.activate_ability()

        target = self.board.at(Point(0, 4)) if self.board.at(Point(0, 4)) is not None else self.board.at(Point(0, 2))
        self.assertEqual(target.card_id, "b006")
        self.assertEqual(target.strength, 1)