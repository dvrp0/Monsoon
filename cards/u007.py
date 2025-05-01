from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U007(Unit): # Green Prototypes
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.CONSUTRUCT], 1, 5, 1, TriggerType.ON_DEATH)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_surrounding_tiles(
            Context(self.position, pov=self.player, source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        if len(targets) > 0:
            target = self.player.board.at(self.player.random.choice(targets))
            target.heal(self.ability_strength)
            target.vitalize()

class U007Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 6)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 5)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 5)
        self.board.spawn_token_unit(self.local, Point(1, 3), 5)

        card = U007()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, 6)
        self.assertTrue(self.board.at(Point(0, 3)).is_vitalized)
        self.assertEqual(self.board.at(Point(0, 2)).strength, 5)
        self.assertEqual(self.board.at(Point(1, 2)).strength, 5)
        self.assertEqual(self.board.at(Point(1, 3)).strength, 5)

        self.board.clear()
        self.board.spawn_token_unit(self.local, Point(1, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 1)

        card = U007()
        card.player = self.local
        card.play(Point(2, 3))

        card_remote = U007()
        card_remote.player = self.remote
        card_remote.play(Point(3, 2))

        self.assertEqual(self.board.at(Point(2, 2)), None)
        self.assertEqual(self.board.at(Point(3, 2)), None)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(2, 0), 5)
        self.board.spawn_token_unit(self.remote, Point(2, 1), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 1), 1)
        self.board.spawn_token_unit(self.local, Point(1, 2), 3)
        self.board.spawn_token_unit(self.local, Point(1, 3), 3)
        card = U007()
        card.player = self.local
        card.play(Point(2, 2))

        self.assertEqual(self.board.at(Point(1, 1)).strength, 1)
        self.assertFalse(self.board.at(Point(1, 1)).is_vitalized)