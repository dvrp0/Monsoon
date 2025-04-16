from enums import Faction, StatusEffect
from point import Point
from target import Target
from structure import Structure
from unit import Unit
from test import CardTestCase
from .u020 import U020

class B008(Structure): # Temple of the Mind
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3, 9)

    def activate_ability(self, position: Point | None = None):
        if len(self.player.hand) > 0 and isinstance(self.player.hand[0], Unit):
            self.player.hand[0].fixedly_forward = not self.player.hand[0].fixedly_forward

        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ANY, status_effects=[StatusEffect.CONFUSED]))

        if len(targets) > 0:
            targets.sort(key=lambda t: (self.player.board.at(t).strength, self.player.random.random()))
            self.player.board.at(targets[0]).destroy()

class B008Test(CardTestCase):
    def test_ability(self):
        card = B008()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(card.player.hand[0].fixedly_forward, True)

        self.board.spawn_token_unit(self.remote, Point(1, 4), 10)
        self.board.spawn_token_unit(self.local, Point(3, 4), 99).confuse()
        card.activate_ability()

        self.assertEqual(card.player.hand[0].fixedly_forward, False)
        self.assertIsNotNone(self.board.at(Point(1, 4)))
        self.assertIsNone(self.board.at(Point(3, 4)))

        u020 = U020()
        u020.player = self.local
        self.local.hand[0] = u020
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 0), 1)
        s1 = self.board.spawn_token_structure(self.remote, Point(1, 1), 1)
        card.activate_ability()
        self.local.hand[0].play(Point(0, 1))

        self.assertEqual(self.remote.strength, 20 - U020().strength)
        self.assertEqual(u1.strength, 1)
        self.assertEqual(s1.strength, 1)