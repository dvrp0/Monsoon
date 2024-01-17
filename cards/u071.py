from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U071(Unit): # Angelic Tikas
    def __init__(self):
        super().__init__([UnitType.FELINE, UnitType.ANCIENT], 3, 6, 1, TriggerType.BEFORE_MOVING)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY), self.player)
        not_confused = [tile for tile in targets if not self.player.board.at(tile).is_confused]

        if len(not_confused) > 0:
            self.player.board.at(self.player.random.choice(not_confused)).confuse()
            front = self.player.board.get_front_tiles(self.position)

            if len(front) > 0 and self.player.board.at(front[0]) is None:
                self.teleport(front[0])
                self.set_path(True)

class U071Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        card = U071()
        card.player = self.local
        card.play(Point(1, 1))

        self.assertTrue(self.board.at(Point(0, 1)).is_confused)
        self.assertEqual(self.remote.strength, 14)

        card = U071()
        card.player = self.local
        card.play(Point(1, 1))

        self.assertEqual(self.board.at(Point(0, 1)).card_id, "u071")
        self.assertEqual(self.board.at(Point(0, 1)).strength, 5)

        card = U071()
        card.player = self.local
        card.play(Point(2, 1))

        self.assertEqual(self.board.at(Point(2, 0)).card_id, "u071")

        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 5)
        card = U071()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.board.at(Point(2, 2)).card_id, "u071")
        self.assertEqual(self.board.at(Point(2, 2)).strength, 1)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 3), 3)
        self.board.spawn_token_unit(self.local, Point(0, 2), 3)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 9)
        card = U071()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(self.board.at(Point(1, 0)).card_id, "u071")
        self.assertEqual(self.local.front_line, 1)