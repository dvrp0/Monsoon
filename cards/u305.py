from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U305(Unit): # Linked Golems
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 3, 3, 1, TriggerType.ON_PLAY)
        self.ability_strength = 4

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.CONSUTRUCT]))

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)
            self.heal(self.ability_strength)

class U305Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 4), 5, [UnitType.CONSUTRUCT])
        self.board.spawn_token_unit(self.local, Point(2, 4), 5, [UnitType.CONSUTRUCT])

        card = U305()
        card.player = self.local
        card.play(Point(1, 4))

        self.assertTrue(self.board.at(Point(0, 4)).strength == 5 + card.ability_strength or self.board.at(Point(2, 4)).strength == 5 + card.ability_strength)
        self.assertEqual(card.strength, U305().strength + card.ability_strength)