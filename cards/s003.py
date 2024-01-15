from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S003(Spell): # Bladestorm
    def __init__(self):
        super().__init__(5)

    def activate_ability(self, position: Point | None = None):
        for tile in self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY)):
            self.player.board.at(tile).deal_damage(self.player.random.randint(4, 5))

class S003Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.local, Point(0, 4), 8)
        self.board.spawn_token_unit(self.local, Point(0, 3), 3)
        self.board.spawn_token_structure(self.remote, Point(1, 4), 4)
        self.board.spawn_token_structure(self.remote, Point(2, 4), 4)
        self.board.spawn_token_unit(self.remote, Point(3, 4), 4)
        self.board.spawn_token_unit(self.remote, Point(3, 2), 6)
        card = S003()
        card.player = self.local
        card.play()

        self.assertEqual(self.board.at(Point(0, 4)).strength, 8)
        self.assertEqual(self.board.at(Point(0, 3)).strength, 3)
        self.assertEqual(self.board.at(Point(1, 4)), None)
        self.assertEqual(self.board.at(Point(2, 4)), None)
        self.assertEqual(self.board.at(Point(3, 4)), None)
        self.assertLessEqual(self.board.at(Point(3, 2)).strength, 2)
        self.assertEqual(self.remote.front_line, 2)