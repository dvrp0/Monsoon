from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U306(Unit): # Destuctobots
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 2, 6, 1, TriggerType.ON_PLAY)
        self.ability_damage = 1

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.FRIENDLY), self.position)

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).deal_damage(self.ability_damage, source=self)

class U306Test(CardTestCase):
    def test_ability(self):
        card = U306()
        card.player = self.local
        self.board.spawn_token_unit(self.remote, Point(3, 1), 5)
        self.board.spawn_token_structure(self.local, Point(0, 4), card.ability_damage + 1)
        self.board.spawn_token_unit(self.local, Point(1, 4), card.ability_damage + 1)
        card.play(Point(2, 4))

        self.assertEqual(self.board.at(Point(3, 1)).strength, 5)
        self.assertTrue(self.board.at(Point(0, 4)).strength == 1 or self.board.at(Point(1, 4)).strength == 1)