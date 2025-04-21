from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UE01(Unit): # Trekking Aldermen
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ELDER], 3, 8, 0, TriggerType.AFTER_SURVIVING)

    def activate_ability(self, position: Point | None = None):
        for _ in range(self.damage_taken):
            targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ENEMY), perspective=self.player)

            if len(targets) > 0:
                self.player.board.at(self.player.random.choice(targets)).deal_damage(1)

class UE01Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 0), 1)
        u2 = self.board.spawn_token_unit(self.remote, Point(1, 0), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(2, 0), 1)
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 0), 2)
        u5 = self.board.spawn_token_unit(self.remote, Point(2, 1), 1)
        u6 = self.board.spawn_token_unit(self.local, Point(2, 4), 1)
        s1 = self.board.spawn_token_structure(self.remote, Point(3, 2), 1)
        card = UE01()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(6)

        self.assertEqual(u1.strength + u2.strength + u3.strength + u4.strength + u5.strength + \
            u6.strength + s1.strength, 2)

        self.board.clear()
        strength = UE01().strength - 1
        u1 = self.board.spawn_token_unit(self.remote, Point(3, 0), strength)
        card = UE01()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(card.strength)

        self.assertEqual(u1.strength, strength)

        card = UE01()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(card.strength - 1)

        self.assertEqual(u1.strength, 0)