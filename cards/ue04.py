from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UE04(Unit): # Greenwood Ancients
    def __init__(self):
        super().__init__([UnitType.ELDER], 4, 6, 1, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ENEMY), perspective=self.player)
        self.heal(sum(self.player.board.at(target).strength > self.strength for target in targets) * 4)

class UE04Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 0), 2)
        self.board.spawn_token_unit(self.remote, Point(2, 0), 7)
        self.board.spawn_token_unit(self.remote, Point(3, 0), 12)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 7)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        card = UE04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, 17)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 0), 3)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 3)
        card = UE04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, 3)