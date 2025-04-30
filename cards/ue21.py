from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UE21(Unit): # Petrified Fossils
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD, UnitType.ELDER], 4, 8, 1, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, strength_limit=self.strength),
            self.position, self.player)

        for target in targets:
            self.player.board.at(target).command()

class UE21Test(CardTestCase):
    def test_ability(self):
        card = UE21()
        card.player = self.local
        self.board.spawn_token_structure(self.remote, Point(2, 3), 1)
        self.board.spawn_token_structure(self.local, Point(1, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(3, 0), 1)
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), card.strength - 1)
        u2 = self.board.spawn_token_unit(self.local, Point(0, 0), card.strength)
        u3 = self.board.spawn_token_unit(self.local, Point(1, 3), 1)
        u4 = self.board.spawn_token_unit(self.local, Point(1, 1), 2)
        u5 = self.board.spawn_token_unit(self.remote, Point(3, 4), 1)
        u6 = self.board.spawn_token_unit(self.local, Point(3, 3), 1)
        self.board.spawn_token_unit(self.local, Point(2, 0), card.strength - 1)
        card.play(Point(3, 1))

        self.assertEqual(u1.position, Point(0, 3))
        self.assertEqual(u2.position, Point(0, 0))
        self.assertEqual(u3.position, Point(1, 3))
        self.assertEqual(u4.position, Point(1, 0))
        self.assertEqual(u5.position, Point(3, 4))
        self.assertEqual(u6.position, Point(3, 2))
        self.assertEqual(self.remote.strength, 20 - (UE21().strength - 1))