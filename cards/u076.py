from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U076(Unit): # Divine Reptiles; DVRP🥰
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.DRAGON, UnitType.ELDER], 5, 9, 1, TriggerType.AFTER_SURVIVING)
        self.ability_damage = 6
        self.ability_strength = 6

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY, exclude_unit_types=[UnitType.DRAGON]))

        if len(targets) > 0:
            target = self.player.board.at(self.player.random.choice(targets))
            target.deal_damage(self.ability_damage)

            if target.strength <= 0:
                self.player.board.spawn_token_unit(self.player, target.position, self.ability_strength, [UnitType.DRAGON])

class U076Test(CardTestCase):
    def test_ability(self):
        card = U076()
        card.player = self.local
        self.board.spawn_token_unit(self.local, Point(1, 4), card.ability_damage, [UnitType.PIRATE])
        u1 = self.board.spawn_token_unit(self.remote, Point(1, 3), 1, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(1, 4)).strength, card.ability_strength)
        self.assertEqual(self.board.at(Point(1, 4)).unit_types, [UnitType.DRAGON])
        self.assertEqual(u1.strength, 1)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        card = U076()
        card.player = self.local
        card.play(Point(0, 4))

        units = self.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY), perspective=self.local)
        self.assertEqual(len(units), 1)