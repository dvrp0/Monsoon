from card import Card
from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U320(Unit): # Original Blueprints
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT, UnitType.ANCIENT], 4, 7, 1, TriggerType.BEFORE_MOVING)
        self.ability_strength = 4

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY), self.player)

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)

class U320Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 1)
        self.board.spawn_token_unit(self.local, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        card = U320()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertTrue(self.board.at(Point(0, 4)).strength == 1 + card.ability_strength or
            self.board.at(Point(1, 4)).strength == 1 + card.ability_strength)
        self.assertEqual(self.board.at(Point(2, 4)).strength, 1)