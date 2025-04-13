from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class U050(Unit): # Twilight Prowlers
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 6, 15, 0, TriggerType.ON_PLAY)
        self.ability_movement = 3

    def activate_ability(self, position: Point | None = None):
        if self.position.y == 4:
            self.gain_speed(self.ability_movement)

class U050Test(CardTestCase):
    def test_ability(self):
        card = U050()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 4 - card.ability_movement))

        card = U050()
        card.player = self.local
        card.play(Point(0, 3))

        self.assertEqual(card.position, Point(0, 3))