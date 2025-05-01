from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class UE05(Unit): # Prime Oracle Bragda
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ELDER], 6, 11, 1, TriggerType.AFTER_SURVIVING)
        self.ability_amount = 4

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(exclude=self.position, pov=self.player, source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY, strength_limit=self.strength - 1)
        )

        for target in targets[:self.ability_amount]:
            self.player.board.at(target).strength = self.strength

class UE05Test(CardTestCase):
    def test_ability(self):
        card = UE05()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.local, Point(1, 3), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(1, 4), 3)
        u3 = self.board.spawn_token_unit(self.local, Point(3, 4), card.strength - 5)
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 0), 1)
        u5 = self.board.spawn_token_unit(self.local, Point(2, 1), 99)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        s1 = self.board.spawn_token_structure(self.local, Point(2, 2), 1)
        card.play(Point(0, 4))

        self.assertEqual(u1.strength, card.strength)
        self.assertEqual(u2.strength, card.strength)
        self.assertEqual(u3.strength, card.strength)
        self.assertEqual(u4.strength, 1)
        self.assertEqual(u5.strength, 99)
        self.assertEqual(s1.strength, 1)