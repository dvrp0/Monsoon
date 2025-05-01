from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class UE22(Unit): # Bucks of Wasteland
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR, UnitType.ELDER], 6, 14, 0, TriggerType.AFTER_SURVIVING)
        self.ability_amount = 2

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(exclude=self.position, pov=self.player, source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        )

        if len(targets) > 0:
            self.player.random.shuffle(targets)

            for target in targets[:self.ability_amount]:
                self.player.board.at(target).heal(self.damage_taken)

class UE22Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(3, 4), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(2, 3), 2)
        u3 = self.board.spawn_token_unit(self.local, Point(2, 2), 3)
        u4 = self.board.spawn_token_unit(self.remote, Point(2, 0), 5)
        card = UE22()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(10)

        self.assertEqual(card.strength, UE22().strength - 10)
        self.assertEqual(u1.strength + u2.strength + u3.strength, 26)
        self.assertEqual(u4.strength, 5)

        card.destroy()

        self.assertEqual(u1.strength + u2.strength + u3.strength, 26)
        self.assertEqual(u4.strength, 5)