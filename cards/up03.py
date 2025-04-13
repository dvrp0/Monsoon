from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UP03(Unit): # Terrifying Behemoths
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PRIMAL, UnitType.HERO], 6, 12, 1, TriggerType.ON_PLAY)
        self.ability_amount = 3

    def activate_ability(self, position: Point | None = None):
        types = []
        for target in self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY)):
            types += self.player.board.at(target).unit_types
        types = list(set(types))

        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY))

        for target in targets:
            self.player.board.at(target).reduce(self.ability_amount)

class UP03Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(3, 1), 1, [UnitType.PRIMAL])
        self.board.spawn_token_unit(self.local, Point(3, 3), 1, [UnitType.SATYR])
        self.board.spawn_token_unit(self.local, Point(1, 1), 1, [UnitType.SATYR, UnitType.ANCIENT])
        self.board.spawn_token_unit(self.remote, Point(2, 4), 99, [UnitType.CONSUTRUCT, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(0, 4), 99, [UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1, [UnitType.PIRATE])
        self.board.spawn_token_structure(self.remote, Point(2, 2), 5)
        card = UP03()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 99 - card.ability_amount)
        self.assertEqual(self.board.at(Point(2, 4)).strength, 99 - card.ability_amount)
        self.assertEqual(self.board.at(Point(0, 3)).strength, UP03().strength - 1)
        self.assertEqual(self.board.at(Point(2, 2)).strength, 5)