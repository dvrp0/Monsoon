from enums import Faction, TriggerType, UnitType
from point import Point
from unit import Unit
from test import CardTestCase

class U036(Unit): # Northsea Dog
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.PIRATE], 2, 1, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        if len(self.player.hand) == 0:
            self.player.board.at(self.position).heal(12)

class U036Test(CardTestCase):
    def test_ability(self):
        self.local.discard(self.local.hand[0])
        card = U036()
        card.player = self.local
        self.local.hand.append(card)
        self.local.play(3, Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 1)

        self.local.hand = []
        card = U036()
        card.player = self.local
        self.local.hand.append(card)
        self.local.play(0, Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 4)).strength, 13)