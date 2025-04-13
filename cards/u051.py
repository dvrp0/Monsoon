from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U051(Unit): # Razor-sharp Lynxes
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 4, 6, 1, TriggerType.ON_PLAY)
        self.ability_movement = 1
        self.ability_strength = 2

    def activate_ability(self, position: Point | None = None):
        if len(self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ANY))) == 0:
            self.gain_speed(self.ability_movement)
        else:
            self.heal(self.ability_strength)

class U051Test(CardTestCase):
    def test_ability(self):
        card = U051()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4 - card.movement - card.ability_movement))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 3), 5)
        card = U051()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 3))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 4), 5)
        card = U051()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 4))
        self.assertEqual(card.strength, U051().strength + card.ability_strength - 5)