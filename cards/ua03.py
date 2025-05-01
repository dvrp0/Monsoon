from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class UA03(Unit): # Lost Psyches
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ANCIENT], 3, 7, 1, TriggerType.BEFORE_MOVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        tiles = [tile for tile in self.player.board.get_row_tiles(Context(self.position, source=self))
            if self.player.board.at(tile) is None] + [self.position]

        self.teleport(self.player.random.choice(tiles))

class UA03Test(CardTestCase):
    def test_ability(self):
        card = UA03()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(card.position.x in [0, 1, 2, 3])

        self.board.clear()
        self.board.spawn_token_structure(self.local, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(3, 4), 1)
        card = UA03()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position.x, 0)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        self.board.spawn_token_unit(self.local, Point(2, 4), 1)
        card = UA03()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(card.position == Point(1, 4) or card.position == Point(3, 3))