from card import Card
from enums import Faction
from point import Point
from spell import Spell
from target import Context, Target
from test import CardTestCase
from typing import List

class S203(Spell): # Dark Harvest
    def __init__(self):
        super().__init__(Faction.SWARM, 5)
        self.ability_damage = 6

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        tiles: List[Point] = []
        friendlies = self.player.board.get_targets(
            Context(source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        )

        for friendly in friendlies:
            tiles += self.player.board.get_surrounding_tiles(
                Context(friendly, source=self),
                Target(Target.Kind.ANY, Target.Side.ENEMY, include_base=True)
            )

        tiles = list(set(tiles)) 

        for tile in tiles:
            self.player.board.at(tile).deal_damage(self.ability_damage, source=self)

class S203Test(CardTestCase):
    def test_ability(self):
        card = S203()
        card.player = self.local
        self.board.spawn_token_unit(self.remote, Point(0, 3), 5)
        self.board.spawn_token_structure(self.remote, Point(1, 3), card.ability_damage - 1)
        self.board.spawn_token_unit(self.remote, Point(3, 3), card.ability_damage - 1)
        self.board.spawn_token_unit(self.remote, Point(2, 1), card.ability_damage - 1)
        self.board.spawn_token_structure(self.remote, Point(1, 0), card.ability_damage + 4)
        self.board.spawn_token_unit(self.local, Point(2, 3), 1)
        self.board.spawn_token_unit(self.local, Point(2, 2), 1)
        self.board.spawn_token_unit(self.local, Point(1, 1), 1)
        self.board.spawn_token_unit(self.local, Point(3, 0), 1)
        card.play()

        self.assertEqual(self.board.at(Point(0, 3)).strength, 5)
        self.assertEqual(self.board.at(Point(1, 3)), None)
        self.assertEqual(self.board.at(Point(3, 3)), None)
        self.assertEqual(self.board.at(Point(2, 1)), None)
        self.assertEqual(self.board.at(Point(1, 0)).strength, 4)
        self.assertEqual(self.remote.strength, 14)