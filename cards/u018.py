from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U018(Unit): # Ubass the Hunter
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PRIMAL, UnitType.HERO], 5, 12, 0, TriggerType.ON_PLAY)
        self.ability_damage = 2

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))
        types = list(set([self.player.board.at(target).unit_types[0] for target in targets]))

        for _ in range(len(types)):
            targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY), include_base=True)
            self.player.board.at(self.player.random.choice(targets)).deal_damage(self.ability_damage, source=self)

class U018Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(3, 1), 1, [UnitType.PRIMAL])
        self.board.spawn_token_unit(self.local, Point(0, 3), 1, [UnitType.SATYR])
        self.board.spawn_token_unit(self.local, Point(1, 4), 1, [UnitType.SATYR, UnitType.ANCIENT])
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1, [UnitType.CONSUTRUCT, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(0, 4), 1, [UnitType.HERO])
        card = U018()
        card.player = self.local
        card.play(Point(1, 3))

        damages = (0 if self.board.at(Point(0, 4)) else card.ability_damage) + \
            (0 if self.board.at(Point(2, 4)) else card.ability_damage) + (20 - self.remote.strength)
        self.assertEqual(damages, card.ability_damage * 3)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(0, 4), 1, [UnitType.PRIMAL])
        self.board.spawn_token_unit(self.local, Point(0, 3), 1, [UnitType.ANCIENT])
        self.board.spawn_token_unit(self.local, Point(0, 2), 1, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.local, Point(1, 2), 1, [UnitType.ELDER])
        self.board.spawn_token_unit(self.local, Point(2, 2), 1, [UnitType.FELINE])
        self.board.spawn_token_unit(self.local, Point(2, 3), 1, [UnitType.RODENT])
        self.board.spawn_token_unit(self.local, Point(2, 4), 1, [UnitType.PIRATE])
        self.board.spawn_token_unit(self.local, Point(1, 4), 1, [UnitType.FLAKE])
        strength_cache = 14
        self.board.spawn_token_structure(self.remote, Point(2, 1), strength_cache)
        self.board.spawn_token_unit(self.remote, Point(3, 2), strength_cache)
        self.remote.strength = 20
        card = U018()
        card.player = self.local
        card.play(Point(1, 3))

        first = self.board.at(Point(2, 1))
        second = self.board.at(Point(3, 2))
        damages = (strength_cache - first.strength if first else strength_cache) + \
            (strength_cache - second.strength if second else strength_cache) + (20 - self.remote.strength)
        self.assertEqual(damages, card.ability_damage * 8)