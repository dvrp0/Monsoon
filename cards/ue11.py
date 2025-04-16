from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class UE11(Unit): # Earthfathers
    def __init__(self):
        super().__init__(Faction.WINTER, [UnitType.VIKING, UnitType.ELDER], 7, 12, 1, TriggerType.AFTER_SURVIVING)
        self.ability_strength = 6

    def activate_ability(self, position: Point | None = None):
        self.heal(self.ability_strength)

class UE11Test(CardTestCase):
    def test_ability(self):
        card = UE11()
        card.player = self.local
        strength = card.strength
        card.play(Point(0, 4))
        card.deal_damage(2)

        self.assertEqual(card.strength, strength - 2 + card.ability_strength)

        strength = card.strength
        card.deal_damage(10)
        self.assertEqual(card.strength, strength - 10 + card.ability_strength)

        card.destroy()

        self.assertIsNone(self.board.at(Point(0, 3)))