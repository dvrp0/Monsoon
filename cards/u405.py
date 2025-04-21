from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U405(Unit): # Witches of the Wild
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.RAVEN], 4, 5, 1, TriggerType.ON_PLAY)
        self.ability_strength = 3

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))

        for target in targets:
            dealt = self.player.board.at(target).deal_damage(self.ability_strength, source=self)
            self.heal(dealt)

class U405Test(CardTestCase):
    def test_ability(self):
        card = U405()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 4), card.ability_strength - 1)
        u2 = self.board.spawn_token_unit(self.local, Point(2, 4), card.ability_strength)
        self.board.spawn_token_structure(self.local, Point(1, 3), 3)
        card.play(Point(1, 4))

        self.assertLessEqual(u1.strength, 0)
        self.assertLessEqual(u2.strength, 0)
        self.assertEqual(card.strength, U405().strength + card.ability_strength * 2 - 1)