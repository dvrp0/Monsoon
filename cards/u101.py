from enums import Faction, UnitType, TriggerType, StatusEffect
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U101(Unit): # Wisp Cloud
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE], 3, 6, 1, TriggerType.BEFORE_ATTACKING)
        self.ability_damage = 8

    def activate_ability(self, position: Point | None = None):
        # When attacking base, structure, or unfrozen unit
        if not position.is_valid or not isinstance(self.player.board.at(position), Unit) or \
            not self.player.board.at(position).is_frozen:
            return

        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY,
            status_effects=[StatusEffect.FROZEN]))

        if len(targets) > 0:
            for target in targets:
                self.player.board.at(target).deal_damage(self.ability_damage)

class U101Test(CardTestCase):
    def test_ability(self):
        card = U101()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 4), card.ability_damage);
        u1.freeze()
        u2 = self.board.spawn_token_unit(self.remote, Point(2, 3), card.ability_damage);
        u2.freeze()
        u3 = self.board.spawn_token_unit(self.remote, Point(3, 3), card.ability_damage);
        u3.freeze()
        u4 = self.board.spawn_token_unit(self.remote, Point(3, 4), 5);
        u5 = self.board.spawn_token_unit(self.local, Point(1, 3), 4);
        card.play(Point(2, 4))

        self.assertEqual(card.strength, U101().strength)
        self.assertLessEqual(u1.strength, 0)
        self.assertLessEqual(u2.strength, 0)
        self.assertLessEqual(u3.strength, 0)
        self.assertEqual(u4.strength, 5)
        self.assertEqual(u5.strength, 4)

        self.board.clear()
        u1 = self.board.spawn_token_unit(self.remote, Point(2, 3), 20);
        u2 = self.board.spawn_token_unit(self.remote, Point(3, 3), 5);
        u2.freeze()
        u3 = self.board.spawn_token_unit(self.remote, Point(3, 4), 5);
        u3.freeze()
        card = U101()
        card.player = self.local
        card.play(Point(2, 4))

        self.assertEqual(u1.strength, 20 - U101().strength)
        self.assertEqual(u2.strength + u3.strength, 10)