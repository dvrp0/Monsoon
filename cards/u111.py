from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U111(Unit): # Snowmasons
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.VIKING], 4, 2, 1, TriggerType.ON_DEATH)
        self.ability_strength = 10

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        for _ in range(self.ability_strength):
            targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY), self.player)

            if len(targets) > 0:
                self.player.board.at(self.player.random.choice(targets)).heal(1)

class U111Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(1, 4), 1)
        u3 = self.board.spawn_token_unit(self.local, Point(2, 4), 1)
        u4 = self.board.spawn_token_unit(self.remote, Point(0, 3), 5)
        u5 = self.board.spawn_token_unit(self.remote, Point(2, 3), 5)
        s1 = self.board.spawn_token_structure(self.remote, Point(1, 2), 5)
        s2 = self.board.spawn_token_structure(self.local, Point(2, 2), 1)
        card = U111()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(u1.strength + u2.strength + u3.strength, 3 + card.ability_strength)
        self.assertEqual(u4.strength, 5)
        self.assertEqual(u5.strength, 5)
        self.assertEqual(s1.strength, 5 - U111().strength)
        self.assertEqual(s2.strength, 1)