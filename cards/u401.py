from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U401(Unit): # Crimson Sentry
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.TOAD], 3, 1, 2, TriggerType.ON_DEATH)
        self.damage = 5

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))

        for target in targets:
            unit = self.player.board.at(target)

            if unit:
                unit.deal_damage(self.damage, source=self)
                unit.poison()

class U401Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(1, 0), 8)
        self.board.spawn_token_unit(self.local, Point(0, 1), 5)
        self.board.spawn_token_unit(self.remote, Point(2, 1), 10)
        card = U401()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(self.board.at(Point(1, 0)).strength, 7)
        self.assertEqual(self.board.at(Point(0, 1)), None)
        self.assertEqual(self.board.at(Point(2, 1)).strength, 5)
        self.assertTrue(self.board.at(Point(2, 1)).is_poisoned)
        self.assertEqual(self.board.at(Point(1, 1)), None)
        self.assertEqual(self.board.at(Point(1, 2)), None)