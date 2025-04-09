from enums import Faction, UnitType
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S013(Spell): # Hunter's Vengeance
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3)

    def activate_ability(self, position: Point | None = None):
        targets = []

        for unit_type in list(UnitType):
            units = [unit for unit in self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ANY, [unit_type]))
                     if unit not in targets]

            if len(units) > 0:
                targets.append(self.player.random.choice(units))

        for target in targets:
            self.player.board.at(target).deal_damage(6)

class S013Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 7, [UnitType.SATYR, UnitType.HERO])
        self.board.spawn_token_unit(self.local, Point(1, 4), 7, [UnitType.SATYR, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(2, 4), 7, [UnitType.FLAKE])
        self.board.spawn_token_unit(self.remote, Point(3, 4), 7, [UnitType.FLAKE])
        self.board.spawn_token_unit(self.remote, Point(0, 3), 6, [UnitType.ANCIENT])
        self.board.spawn_token_unit(self.local, Point(1, 3), 6, [UnitType.CONSUTRUCT])
        self.board.spawn_token_unit(self.remote, Point(2, 3), 6, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.local, Point(3, 3), 6, [UnitType.ELDER])
        card = S013()
        card.player = self.local
        card.play()

        self.assertEqual(self.board.at(Point(0, 4)).strength, 1)
        self.assertEqual(self.board.at(Point(1, 4)).strength, 1)
        self.assertTrue(self.board.at(Point(2, 4)).strength == 1 or self.board.at(Point(3, 4)).strength == 1)
        self.assertEqual(self.board.at(Point(0, 3)), None)
        self.assertEqual(self.board.at(Point(1, 3)), None)
        self.assertEqual(self.board.at(Point(2, 3)), None)
        self.assertEqual(self.board.at(Point(3, 3)), None)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 2), 6, [UnitType.ANCIENT])
        self.board.spawn_token_unit(self.remote, Point(3, 2), 6, [UnitType.ELDER])
        card.play()

        self.assertEqual(self.board.at(Point(0, 2)), None)
        self.assertEqual(self.board.at(Point(3, 2)), None)
        self.assertEqual(self.local.front_line, 2)
        self.assertEqual(self.remote.front_line, 0)