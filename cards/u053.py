from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U053(Unit): # Wild Saberpaws
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 2, 5, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        if len(self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))) == 0:
            self.gain_speed(2)
        elif len(self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))) == 0:
            self.gain_speed(1)

class U053Test(CardTestCase):
    def test_ability(self):
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 3), 5)
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 3))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        card = U053()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4))

        self.board.clear()
        self.board.spawn_token_structure(self.remote, Point(1, 3), 1)
        card = U053()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(card.position, Point(1, 0))