from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class UE42(Unit): # Hairy Chesetnuts
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.TOAD, UnitType.ELDER], 6, 14, 0, TriggerType.AFTER_SURVIVING)
        self.ability_damage = 2

    def activate_ability(self, position: Point | None = None):
        amount = min(self.ability_damage, self.damage_taken)
        self.player.opponent.deal_damage(amount)
        self.heal(amount)

class UE42Test(CardTestCase):
    def test_ability(self):
        card = UE42()
        card.player = self.local
        card.play(Point(0, 4))
        card.deal_damage(5)

        strength = UE42().strength - 5 + card.ability_damage
        base_strength = 20 - card.ability_damage
        self.assertEqual(card.strength, strength)
        self.assertEqual(self.remote.strength, base_strength)

        card.deal_damage(1)

        base_strength -= 1
        self.assertEqual(card.strength, strength)
        self.assertEqual(self.remote.strength, base_strength)

        card.destroy()

        self.assertEqual(self.remote.strength, base_strength)