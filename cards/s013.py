from card import Card
from enums import Faction, UnitType
from point import Point
from spell import Spell
from target import Context, Target
from test import CardTestCase

class S013(Spell): # Hunter's Vengeance
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3)
        self.ability_damage = 6

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = []

        for unit_type in list(UnitType):
            units = [unit for unit in self.player.board.get_targets(
                Context(source=self),
                Target(Target.Kind.UNIT, Target.Side.ANY, [unit_type])
            ) if unit not in targets]

            if len(units) > 0:
                targets.append(self.player.random.choice(units))

        for target in targets:
            self.player.board.at(target).deal_damage(self.ability_damage, source=self)

class S013Test(CardTestCase):
    def test_ability(self):
        card = S013()
        card.player = self.local
        self.board.spawn_token_unit(self.local, Point(0, 4), card.ability_damage + 1, [UnitType.SATYR, UnitType.HERO])
        self.board.spawn_token_unit(self.local, Point(1, 4), card.ability_damage + 1, [UnitType.SATYR, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(2, 4), card.ability_damage + 1, [UnitType.FLAKE])
        self.board.spawn_token_unit(self.remote, Point(3, 4), card.ability_damage + 1, [UnitType.FLAKE])
        self.board.spawn_token_unit(self.remote, Point(0, 3), card.ability_damage, [UnitType.ANCIENT])
        self.board.spawn_token_unit(self.local, Point(1, 3), card.ability_damage, [UnitType.CONSUTRUCT])
        self.board.spawn_token_unit(self.remote, Point(2, 3), card.ability_damage, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.local, Point(3, 3), card.ability_damage, [UnitType.ELDER])
        card.play()

        self.assertEqual(self.board.at(Point(0, 4)).strength, 1)
        self.assertEqual(self.board.at(Point(1, 4)).strength, 1)
        self.assertTrue(self.board.at(Point(2, 4)).strength == 1 or self.board.at(Point(3, 4)).strength == 1)
        self.assertEqual(self.board.at(Point(0, 3)), None)
        self.assertEqual(self.board.at(Point(1, 3)), None)
        self.assertEqual(self.board.at(Point(2, 3)), None)
        self.assertEqual(self.board.at(Point(3, 3)), None)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 2), card.ability_damage, [UnitType.ANCIENT])
        self.board.spawn_token_unit(self.remote, Point(3, 2), card.ability_damage, [UnitType.ELDER])
        card.play()

        self.assertEqual(self.board.at(Point(0, 2)), None)
        self.assertEqual(self.board.at(Point(3, 2)), None)
        self.assertEqual(self.local.front_line, 2)
        self.assertEqual(self.remote.front_line, 0)