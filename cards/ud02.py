from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UD02(Unit): # Conflicted Drakes
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.DRAGON], 3, 5, 0, TriggerType.ON_PLAY)
        self.ability_damage = 4

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_front_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))

        for target in targets:
            temp = self.player.board.at(target)

            if UnitType.DRAGON not in temp.unit_types:
                temp.deal_damage(self.ability_damage)

class UD02Test(CardTestCase):
    def test_ability(self):
        card = UD02()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.local, Point(0, 3), card.ability_damage, [UnitType.ELDER])
        u2 = self.board.spawn_token_unit(self.remote, Point(0, 2), 1, [UnitType.DRAGON])
        u3 = self.board.spawn_token_unit(self.remote, Point(0, 1), 1, [UnitType.PIRATE, UnitType.DRAGON])
        s1 = self.board.spawn_token_structure(self.local, Point(0, 0), 1)
        card.play(Point(0, 4))

        self.assertEqual(u1.strength, 0)
        self.assertEqual(u2.strength, 1)
        self.assertEqual(u3.strength, 1)
        self.assertEqual(s1.strength, 1)

        u1 = self.board.spawn_token_unit(self.remote, Point(2, 4), 1, [UnitType.HERO])
        card = UD02()
        card.player = self.local
        card.play(Point(2, 3))

        self.assertEqual(u1.strength, 1)