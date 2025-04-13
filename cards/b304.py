from enums import Faction
from point import Point
from structure import Structure
from test import CardTestCase

class B304(Structure): # Unstable Build
    def __init__(self):
        super().__init__(Faction.IRONCLAD, 2, 9)
        self.ability_strength = 3

    def activate_ability(self, position: Point | None = None):
        self.deal_damage(self.ability_strength)

class B304Test(CardTestCase):
    def test_ability(self):
        card = B304()
        strength_cache = card.strength
        card.player = self.local
        card.play(Point(0, 4))
        card.activate_ability()

        self.assertEqual(card.strength, strength_cache - card.ability_strength)