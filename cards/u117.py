from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U117(Unit): # Iceflakes
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.FLAKE], 2, 8, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        self.freeze()

class U117Test(CardTestCase):
    def test_ability(self):
        card = U117()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(self.board.at(Point(0, 4)).is_frozen)