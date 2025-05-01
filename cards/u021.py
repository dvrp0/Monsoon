from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U021(Unit): # Personal Servers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.CONSUTRUCT], 3, 3, 1, TriggerType.ON_PLAY)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(exclude=self.position, source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        )

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)

class U021Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(3, 4), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        u3 = self.board.spawn_token_unit(self.remote, Point(2, 3), 1)
        s1 = self.board.spawn_token_structure(self.local, Point(2, 1), 1)
        card = U021()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(u1.strength == 1 + card.ability_strength or u2.strength == 1 + card.ability_strength)
        self.assertEqual(u3.strength, 1)
        self.assertEqual(u3.strength, 1)
        self.assertEqual(s1.strength, 1)