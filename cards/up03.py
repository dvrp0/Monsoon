from card import Card
from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Context, Target
from test import CardTestCase

class UP03(Unit): # Terrifying Behemoths
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PRIMAL], 6, 12, 1, TriggerType.ON_PLAY)
        self.ability_amount = 3

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        types = []
        for target in self.player.board.get_targets(
            Context(source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        ):
            types += self.player.board.at(target).unit_types
        types = list(set(types))

        targets = self.player.board.get_surrounding_tiles(
            Context(self.position, source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        for target in targets:
            self.player.board.at(target).reduce(self.ability_amount * len(types))

class UP03Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(3, 1), 1, [UnitType.PRIMAL])
        self.board.spawn_token_unit(self.local, Point(3, 3), 1, [UnitType.SATYR])
        self.board.spawn_token_unit(self.local, Point(1, 1), 1, [UnitType.SATYR, UnitType.ANCIENT])
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 4), 99, [UnitType.HERO])
        u2 = self.board.spawn_token_unit(self.remote, Point(2, 4), 99, [UnitType.CONSUTRUCT, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1, [UnitType.PIRATE])
        s1 = self.board.spawn_token_structure(self.remote, Point(2, 2), 5)
        card = UP03()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(u1.strength, 99 - card.ability_amount * 3)
        self.assertEqual(u2.strength, 99 - card.ability_amount * 3)
        self.assertEqual(card.strength, UP03().strength - 1)
        self.assertEqual(s1.strength, 5)