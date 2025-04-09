from enums import Faction, StatusEffect, UnitType
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S201(Spell): # Catnip's Charm
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 2, Target(Target.Kind.UNIT, Target.Side.ANY, exclude_status_effects=[StatusEffect.CONFUSED]))

    def activate_ability(self, position: Point | None = None):
        self.player.board.at(position).confuse()

        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, unit_types=[UnitType.FELINE]))

        if len(targets) > 0:
            min_strength = min([self.player.board.at(target).strength for target in targets])
            weakests = [target for target in targets if self.player.board.at(target).strength == min_strength]
            self.player.board.at(self.player.random.choice(weakests)).heal(5)

class S201Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 1, [UnitType.FELINE])
        self.board.spawn_token_unit(self.local, Point(1, 4), 1, [UnitType.FELINE])
        self.board.spawn_token_unit(self.local, Point(1, 3), 1, [UnitType.ANCIENT])
        card = S201()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(self.board.at(Point(0, 4)).is_confused)
        self.assertTrue(self.board.at(Point(0, 4)).strength == 6 or self.board.at(Point(1, 4)).strength == 6)
        self.assertEqual(self.board.at(Point(1, 3)).strength, 1)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(2, 4), 3, [UnitType.FELINE])
        self.board.spawn_token_unit(self.local, Point(3, 4), 5, [UnitType.FELINE])
        self.board.spawn_token_unit(self.local, Point(0, 1), 6, [UnitType.FELINE])
        self.board.spawn_token_unit(self.remote, Point(3, 1), 1, [UnitType.FELINE])
        self.board.spawn_token_unit(self.remote, Point(2, 2), 3, [UnitType.FELINE])
        card.play(Point(3, 1))

        self.assertEqual(self.board.at(Point(2, 4)).strength, 8)
        self.assertTrue(self.board.at(Point(3, 1)).is_confused)
        self.assertEqual(self.board.at(Point(3, 1)).strength, 1)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 4), 1, [UnitType.FLAKE])
        self.board.spawn_token_unit(self.remote, Point(3, 4), 1, [UnitType.DRAGON])
        card.play(Point(3, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 1)
        self.assertTrue(self.board.at(Point(3, 4)).is_confused)
        self.assertEqual(self.board.at(Point(3, 4)).strength, 1)