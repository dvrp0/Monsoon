from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class UE12(Unit): # Chilled Stonedames
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE, UnitType.ELDER], 5, 10, 1, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_front_tiles(
            Context(self.position, pov=self.player, source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        for target in targets:
            self.player.board.at(target).destroy(source=self)

class UE12Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(0, 3), 2)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 99)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 10)
        self.board.spawn_token_unit(self.local, Point(0, 0), 5)
        card = UE12()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(1)

        self.assertIsNotNone(self.board.at(Point(0, 3)))
        self.assertIsNone(self.board.at(Point(0, 2)))
        self.assertIsNone(self.board.at(Point(0, 1)))
        self.assertIsNotNone(self.board.at(Point(0, 0)))

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(2, 3), 1)
        self.board.spawn_token_unit(self.local, Point(2, 4), 99)
        card = UE12()
        card.player = self.remote
        self.board.set(Point(2, 2), card)
        self.board.to_next_turn()

        self.assertIsNone(self.board.at(Point(2, 4)))