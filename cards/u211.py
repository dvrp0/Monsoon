from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from test import CardTestCase

class U211(Unit): # Doppelbocks
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.SATYR], 2, 4, 0, TriggerType.ON_PLAY)
        self.ability_min_strength = 1
        self.ability_max_strength = 2

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        front = self.player.board.get_front_tiles(self.position)

        if len(front) > 0 and self.player.board.at(front[-1]) is None:
            self.player.board.spawn_token_unit(self.player, front[-1],
                self.player.random.randint(self.ability_min_strength, self.ability_max_strength + 1), [UnitType.SATYR])

class U211Test(CardTestCase):
    def test_ability(self):
        card = U211()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertLessEqual(self.board.at(Point(0, 3)).strength, card.ability_max_strength)
        self.assertEqual(self.board.at(Point(0, 3)).unit_types, [UnitType.SATYR])
        self.assertEqual(self.local.front_line, 3)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 3), 6)
        card = U211()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(self.board.at(Point(0, 3)).strength, 6)

        self.board.clear()
        self.board.to_next_turn()
        card = U211()
        card.player = self.remote
        card.play(Point(0, 0))

        self.assertIsNotNone(self.board.at(Point(0, 1)))