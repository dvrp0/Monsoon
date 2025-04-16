from enums import Faction
from point import Point
from structure import Structure
from test import CardTestCase

class B007(Structure): # Temple of the Heart
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 3, 7)
        self.ability_strength = 2

    def activate_ability(self, position: Point | None = None):
        if self.player.strength == self.player.opponent.strength:
            return

        stronger = self.player if self.player.strength > self.player.opponent.strength else self.player.opponent
        stronger.deal_damage(self.ability_strength)
        stronger.opponent.heal(self.ability_strength)

class B007Test(CardTestCase):
    def test_ability(self):
        card = B007()
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(self.local.strength, 20)
        self.assertEqual(self.remote.strength, 20)

        self.local.deal_damage(5)
        card.activate_ability()

        self.assertEqual(self.local.strength, 20 - 5 + card.ability_strength)
        self.assertEqual(self.remote.strength, 20 - card.ability_strength)