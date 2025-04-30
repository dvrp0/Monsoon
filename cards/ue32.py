from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UE32(Unit): # Booming Professors
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.RODENT, UnitType.ELDER], 6, 10, 0, TriggerType.AFTER_SURVIVING)
        self.ability_damage = 6

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        damage = min(6, self.strength)
        targets = self.player.board.get_front_tiles(self.position, Target(Target.Kind.ANY, Target.Side.ENEMY), self.player)

        if len(targets) > 0:
            self.player.board.at(targets[-1]).deal_damage(damage, source=self)
        else:
            self.player.opponent.deal_damage(damage)

class UE32Test(CardTestCase):
    def test_ability(self):
        card = UE32()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(5)

        self.assertEqual(self.remote.strength, 20 - min(6, card.strength))

        s1 = self.board.spawn_token_structure(self.remote, Point(0, 1), 1)
        u1 = self.board.spawn_token_unit(self.local, Point(0, 2), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(0, 0), 1)

        card.deal_damage(1)

        self.assertLessEqual(s1.strength, 0)
        self.assertEqual(u1.strength + u2.strength + u3.strength, 3)