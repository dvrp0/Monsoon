from card import Card
from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UA04(Unit): # Gray the Balancer
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ANCIENT, UnitType.HERO], 4, 8, 1, TriggerType.BEFORE_MOVING)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        # no need to set perspective as the result will be the same
        local_units = [self.player.board.at(tile) for tile in self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY))]
        remote_units = [self.player.board.at(tile) for tile in self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.ENEMY))]
        targets = []

        if len(local_units) > len(remote_units):
            strength = min([unit.strength for unit in local_units])
            targets = [target for target in local_units if target.strength == strength]
        elif len(remote_units) > len(local_units):
            strength = min([unit.strength for unit in remote_units])
            targets = [target for target in remote_units if target.strength == strength]

        if len(targets) > 0:
            self.player.random.choice(targets).destroy(source=self)

class UA04Test(CardTestCase):
    def test_ability(self):
        card = UA04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)), None)

        self.board.spawn_token_structure(self.local, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 3), 6)
        self.board.spawn_token_unit(self.remote, Point(1, 3), 5)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        card = UA04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).card_id, "ua04")
        self.assertEqual(self.board.at(Point(0, 3)).strength, 8)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(2, 3), 6)
        card = UA04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).card_id, "ua04")
        self.assertEqual(self.board.at(Point(2, 3)).strength, 6)