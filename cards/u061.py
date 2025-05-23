from unit import Unit
from card import Card
from enums import Faction, UnitType, TriggerType
from point import Point
from target import Context, Target
from test import CardTestCase

class U061(Unit): # Sparkly Kitties
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.FELINE], 2, 6, 0, TriggerType.ON_PLAY)

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        self.confuse()

        units = self.player.board.get_targets(
            Context(exclude=self.position, source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        )
        if len(units) > 0:
            self.player.board.at(self.player.random.choice(units)).confuse()

        self.gain_speed(2)

class U061Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(1, 4), 3)
        card = U061()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 3))
        self.assertEqual(card.strength, 3)

        card = U061()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(1, 4))
        self.assertTrue(self.board.at(Point(1, 3)).is_confused)

        self.board.clear()
        self.board.spawn_token_structure(self.remote, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 4), 1)
        card = U061()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(2, 4))
        self.assertEqual(card.strength, 4)