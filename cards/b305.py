from enums import TriggerType
from point import Point
from structure import Structure
from target import Target
from test import CardTestCase

class B305(Structure): # Temple of Space
    def __init__(self):
        super().__init__(3, 8, [TriggerType.ON_PLAY])
        self.original_cost = self.cost

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))

        if self.position in targets:
            targets.remove(self.position)

        for target in targets:
            if self.player.board.at(target).card_id == self.card_id:
                for tile in self.player.board.get_surrounding_tiles(target, Target(Target.Kind.UNIT, Target.Side.ANY)):
                    to = Point(tile.x - target.x  + self.position.x, tile.y - target.y + self.position.y)

                    if to.is_valid:
                        self.player.board.at(tile).teleport(to)

                self.player.board.at(target).destroy()
                self.player.deck[-1].cost = self.original_cost

                return

        if not self.is_single_use:
            self.player.deck.pop()

        self.cost = 2
        self.weight = 1
        self.player.hand.append(self)

class B305Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(0, 4), 1)
        self.board.spawn_token_structure(self.local, Point(2, 2), 1)
        self.board.spawn_token_unit(self.local, Point(1, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 2)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 3)
        self.board.spawn_token_unit(self.remote, Point(1, 4), 4)
        self.board.spawn_token_unit(self.local, Point(2, 4), 5)
        self.board.spawn_token_unit(self.local, Point(2, 3), 6)

        card = B305()
        card.player = self.local
        card.weight = 1
        self.local.discard(self.local.hand[3])
        self.local.deck[-1] = card
        self.local.hand.append(self.local.deck[-1])
        self.local.deck.pop()
        self.local.play(3, Point(1, 3))

        self.assertEqual(self.local.hand[3].card_id, "b305")
        self.assertEqual(self.local.hand[3].cost, 2)
        self.assertEqual(len(self.local.deck), 8)

        self.local.play(3, Point(2, 1))

        self.assertEqual(self.local.deck[-1].cost, 3)
        self.assertEqual(self.board.at(Point(1, 0)).strength, 2)
        self.assertEqual(self.board.at(Point(2, 0)).strength, 1)
        self.assertEqual(self.board.at(Point(3, 1)).strength, 6)
        self.assertEqual(self.board.at(Point(3, 2)).strength, 5)
        self.assertEqual(self.board.at(Point(1, 1)).strength, 3)
        self.assertEqual(self.board.at(Point(1, 4)).strength, 4)
        self.assertEqual(self.board.at(Point(1, 3)), None)
        self.assertEqual(self.board.at(Point(2, 1)).card_id, "b305")