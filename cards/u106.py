from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U106(Unit): # Hearthguards
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.VIKING], 6, 7, 2, TriggerType.ON_PLAY)
        self.ability_strength = 7

    def activate_ability(self, position: Point | None = None):
        if len(self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))) > 0 or self.position.y == 4:
            self.heal(self.ability_strength)

class U106Test(CardTestCase):
    def test_ability(self):
        card = U106()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, U106().strength + card.ability_strength)

        self.board.clear()
        self.board.spawn_token_structure(self.local, Point(1, 2), 1)
        card = U106()
        card.player = self.local
        card.play(Point(0, 2))

        self.assertEqual(card.strength, U106().strength + card.ability_strength)