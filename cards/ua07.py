from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class UA07(Unit): # Erratic Neglects
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ANCIENT], 1, 5, 0, TriggerType.BEFORE_MOVING)

    def activate_ability(self, position: Point | None = None):
        match self.player.random.randint(0, 4):
            case 0:
                self.freeze()
            case 1:
                self.poison()
            case 2:
                self.vitalize()
            case 3:
                self.confuse()
            case 4:
                self.disable()

class UA07Test(CardTestCase):
    def test_ability(self):
        card = UA07()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertTrue(card.is_frozen or card.is_poisened or card.is_vitalized or card.is_confused or card.is_disabled)