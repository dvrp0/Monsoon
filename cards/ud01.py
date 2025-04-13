from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class UD01(Unit): # Spare Dragonling
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.DRAGON], 2, 1, 1, TriggerType.ON_DEATH)
        self.ability_strength = 7

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY, unit_types=[UnitType.DRAGON]))
        self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)

class UD01Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(2, 4), 1, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.local, Point(3, 1), 1, [UnitType.DRAGON, UnitType.HERO])
        self.board.spawn_token_structure(self.remote, Point(0, 3), 1)
        card = UD01()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(self.board.at(Point(2, 4)).strength == 1 + card.ability_strength or
            self.board.at(Point(3, 1)).strength == 1 + card.ability_strength)