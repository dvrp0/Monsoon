from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from structure import Structure
from target import Target
from test import CardTestCase
from typing import List
from cards.b005 import B005
from cards.b006 import B006
from cards.b203 import B203
from cards.b305 import B305

class UA20(Unit): # Guardi the Lightbringer
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.KNIGHT, UnitType.ANCIENT, UnitType.HERO], 4, 8, 1, TriggerType.BEFORE_MOVING)
        self.ability_cost = 0
        self.ability_level = 5 # unused for now
        self.ability_candidates: List[Structure] = [B005(), B006(), B203(), B305()]

    def activate_ability(self, position: Point | None = None):
        if len(self.player.board.get_front_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY), self.player)) == 0:
            card = self.player.random.choice(self.ability_candidates).copy()
            card.player = self.player
            card.weight = 1
            card.cost = self.ability_cost
            card.is_single_use = True

            self.player.deck.append(card)

class UA20Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 1), 1)
        card = UA20()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(len(self.local.deck), 8)

        self.board.clear()
        self.board.spawn_token_structure(self.remote, Point(0, 1), 1)
        card = UA20()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(len(self.local.deck), 9)
        self.assertEqual(self.local.deck[-1].weight, 1)
        self.assertEqual(self.local.deck[-1].cost, card.ability_cost)
        self.assertTrue(self.local.deck[-1].is_single_use)
        self.assertTrue(self.local.deck[-1].card_id in ["b005", "b006", "b203", "b305"])

        self.local.hand.append(self.local.deck[-1])
        self.local.deck.pop()
        self.local.discard(self.local.hand[3])
        self.local.play(3, Point(0, 4))

        self.assertEqual(len(self.local.deck), 9)