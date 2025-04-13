from enums import Faction, StatusEffect
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S403(Spell): # Soap Cleanse
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, 1)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, status_effects=[StatusEffect.POISONED]))

        if len(targets) > 0:
            for target in targets:
                self.player.board.at(target).heal(self.ability_strength)
                self.player.board.at(target).vitalize()

class S403Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 1)
        self.board.spawn_token_unit(self.local, Point(1, 3), 2)
        self.board.spawn_token_unit(self.local, Point(2, 2), 5)
        self.board.spawn_token_unit(self.local, Point(3, 1), 10)
        self.board.spawn_token_unit(self.remote, Point(0, 4), 5)
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        self.board.at(Point(0, 3)).poison()
        self.board.at(Point(0, 3)).freeze()
        self.board.at(Point(0, 3)).confuse()
        self.board.at(Point(1, 3)).poison()
        self.board.at(Point(1, 3)).freeze()
        self.board.at(Point(3, 1)).poison()
        self.board.at(Point(0, 4)).poison()
        card = S403()
        card.player = self.local
        card.play()

        self.assertEqual(self.board.at(Point(0, 3)).strength, 1 + card.ability_strength)
        self.assertEqual(self.board.at(Point(0, 3)).is_vitalized, True)
        self.assertEqual(self.board.at(Point(1, 3)).strength, 2 + card.ability_strength)
        self.assertEqual(self.board.at(Point(1, 3)).is_vitalized, True)
        self.assertEqual(self.board.at(Point(2, 2)).strength, 5)
        self.assertEqual(self.board.at(Point(2, 2)).is_vitalized, False)
        self.assertEqual(self.board.at(Point(3, 1)).strength, 10 + card.ability_strength)
        self.assertEqual(self.board.at(Point(3, 1)).is_vitalized, True)
        self.assertEqual(self.board.at(Point(0, 4)).strength, 5)
        self.assertEqual(self.board.at(Point(0, 4)).is_poisoned, True)
        self.assertEqual(self.board.at(Point(1, 4)).strength, 5)