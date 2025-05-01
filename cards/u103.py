from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U103(Unit): # Frosthexers
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE], 2, 5, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_bordering_tiles(
            Context(self.position, source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        for target in targets:
            self.player.board.at(target).freeze()

class U103Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.local, Point(0, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 1)
        card = U103()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertFalse(self.board.at(Point(2, 2)).is_frozen)
        self.assertFalse(self.board.at(Point(0, 3)).is_frozen)
        self.assertTrue(self.board.at(Point(1, 4)).is_frozen)
        self.assertTrue(self.board.at(Point(2, 3)).is_frozen)