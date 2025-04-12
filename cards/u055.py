from enums import Faction, StatusEffect, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U055(Unit): # Sweetcap Kittens
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 2, 5, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_front_tiles(self.position,
            Target(Target.Kind.UNIT, Target.Side.ENEMY, exclude_status_effects=[StatusEffect.CONFUSED]))

        for target in targets:
            self.player.board.at(target).confuse()

class U055Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 5)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 4)
        self.board.spawn_token_unit(self.remote, Point(0, 0), 3)
        self.board.spawn_token_unit(self.remote, Point(1, 3), 2)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 1)
        card = U055()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertFalse(self.board.at(Point(0, 3)).is_confused)
        self.assertTrue(self.board.at(Point(0, 2)).is_confused)
        self.assertTrue(self.board.at(Point(0, 1)).is_confused)
        self.assertTrue(self.board.at(Point(0, 0)).is_confused)
        self.assertFalse(self.board.at(Point(1, 3)).is_confused)
        self.assertFalse(self.board.at(Point(1, 2)).is_confused)