from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U316(Unit): # Function Wilds
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 3, 6, 1, TriggerType.ON_PLAY)
        self.ability_amount = 2

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY))

        if len(targets) > 0:
            self.player.random.shuffle(targets)

            for target in targets[:self.ability_amount]:
                self.player.board.at(target).vitalize()

        self.vitalize()

class U316Test(CardTestCase):
    def test_ability(self):
        u3 = self.board.spawn_token_unit(self.local, Point(0, 4), 5)
        u1 = self.board.spawn_token_unit(self.local, Point(1, 4), 5)
        u2 = self.board.spawn_token_unit(self.local, Point(0, 3), 5)
        u4 = self.board.spawn_token_unit(self.remote, Point(2, 2), 5)
        card = U316()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertTrue(card.is_vitalized)
        self.assertEqual(u1.is_vitalized + u2.is_vitalized + u3.is_vitalized + u4.is_vitalized, card.ability_amount)