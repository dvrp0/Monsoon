from enums import UnitType, TriggerType
from point import Point
from spell import Spell
from unit import Unit
from target import Target
from test import CardTestCase
from .s003 import S003
from .s007 import S007

class U017(Unit): # Archdruid Earyn
    def __init__(self):
        super().__init__([UnitType.HERO], 7, 12, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None):
        cards = [card for card in self.player.hand if isinstance(card, Spell) and card.cost <= self.cost]

        if len(cards) > 0:
            self.player.random.shuffle(cards)

            for card in cards[:2]:
                position = None if card.required_targets is None else self.player.random.choice(self.player.board.get_targets(card.required_targets))
                self.player.play(self.player.hand.index(card), position)

class U017Test(CardTestCase):
    def test_ability(self):
        s003 = S003()
        s003.player = self.local
        s007 = S007()
        s007.player = self.local
        self.local.hand = self.local.hand[:2] + [s003, s007]
        self.board.spawn_token_unit(self.remote, Point(3, 3), 4)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 2)
        self.board.spawn_token_unit(self.remote, Point(2, 1), 3)
        self.board.spawn_token_unit(self.remote, Point(0, 0), 4)
        self.board.spawn_token_unit(self.local, Point(3, 4), 5)
        card = U017()
        card.player = self.local
        card.play(Point(1, 4))

        self.assertEqual(len(self.local.hand), 2)
        self.assertTrue(all(x in [card.card_id for card in self.local.deck] for x in ["s003", "s007"]))
        self.assertEqual(self.board.at(Point(3, 3)), None)
        self.assertEqual(self.board.at(Point(0, 1)), None)
        self.assertEqual(self.board.at(Point(2, 1)), None)
        self.assertEqual(self.board.at(Point(0, 0)), None)
        self.assertTrue(self.board.at(Point(1, 4)).strength == 19 or self.board.at(Point(3, 4)).strength == 12)