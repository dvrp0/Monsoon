from card import Card
from enums import Faction
from point import Point
from target import Target
from spell import Spell
from test import CardTestCase

class S101(Spell): # Gift of the Wise
    def __init__(self):
        super().__init__(Faction.WINTER, 9)
        self.ability_mana = 13
        self.ability_strength = 6

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.player.gain_mana(self.ability_mana)

        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY))
        targets.sort(key=lambda t: (self.player.board.at(t).strength, self.player.random.random()))
        self.player.board.at(targets[0]).heal(self.ability_strength)

class S101Test(CardTestCase):
    def test_ability(self):
        u1 = self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        u2 = self.board.spawn_token_unit(self.local, Point(0, 3), 5)
        u3 = self.board.spawn_token_unit(self.local, Point(2, 3), 3)
        u4 = self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        s1 = self.board.spawn_token_structure(self.local, Point(2, 1), 1)
        mana = self.local.current_mana
        card = S101()
        card.player = self.local
        card.play()

        self.assertEqual(self.local.current_mana, mana + card.ability_mana)
        self.assertEqual(u1.strength, 1 + card.ability_strength)
        self.assertEqual(u2.strength, 5)
        self.assertEqual(u3.strength, 3)
        self.assertEqual(u4.strength, 1)
        self.assertEqual(s1.strength, 1)