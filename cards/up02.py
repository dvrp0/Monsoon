from card import Card
from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from target import Context, Target
from test import CardTestCase

class UP02(Unit): # Enranged Cowards
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PRIMAL], 5, 6, 1, TriggerType.ON_PLAY)
        self.ability_strength = 2

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        types = []
        for target in self.player.board.get_targets(
            Context(source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        ):
            types += self.player.board.at(target).unit_types
        types = list(set(types))

        self.heal(self.ability_strength * len(types))

class UP02Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(3, 1), 1, [UnitType.PRIMAL])
        self.board.spawn_token_unit(self.local, Point(3, 3), 1, [UnitType.SATYR])
        self.board.spawn_token_unit(self.local, Point(1, 1), 1, [UnitType.SATYR, UnitType.ANCIENT])
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1, [UnitType.CONSUTRUCT, UnitType.HERO])
        self.board.spawn_token_unit(self.remote, Point(3, 2), 1, [UnitType.HERO])
        card = UP02()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, UP02().strength + card.ability_strength * 3)