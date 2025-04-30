from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Target
from test import CardTestCase

class U310(Unit): # Ozone Purifiers
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.RODENT], 2, 5, 0, TriggerType.ON_PLAY)

    # behind > sides > front
    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_bordering_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.ENEMY))

        behind = next(filter(lambda t: self.position.y + 1 == t.y, targets), None)
        behind_pushable = behind is not None and behind.y < 4 and self.player.board.at(Point(behind.x, behind.y + 1)) is None

        right = next(filter(lambda t: self.position.x + 1 == t.x, targets), None)
        right_pushable = right is not None and right.x < 3 and self.player.board.at(Point(right.x + 1, right.y)) is None

        left = next(filter(lambda t: self.position.x - 1 == t.x, targets), None)
        left_pushable = left is not None and left.x > 0 and self.player.board.at(Point(left.x - 1, left.y)) is None

        front = next(filter(lambda t: self.position.y - 1 == t.y, targets), None)
        front_pushable = front is not None and front.y > 0 and self.player.board.at(Point(front.x, front.y - 1)) is None

        if behind_pushable:
            target = behind
        elif left_pushable:
            target = left
        elif right_pushable:
            target = right
        elif front_pushable:
            target = front

        self.player.board.at(target).push(self.position)

class U310Test(CardTestCase):
    def test_ability(self):
        target = self.board.spawn_token_unit(self.remote, Point(1, 3), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 1), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 1)
        card = U310()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(target.position, Point(1, 4))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 1), 1)
        target = self.board.spawn_token_unit(self.remote, Point(2, 2), 1)
        card = U310()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(target.position, Point(3, 2))

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 2), 1)
        target = self.board.spawn_token_unit(self.remote, Point(1, 1), 1)
        card = U310()
        card.player = self.local
        card.play(Point(1, 2))

        self.assertEqual(target.position, Point(1, 0))

        self.board.clear()
        target = self.board.spawn_token_unit(self.remote, Point(1, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(3, 2), 1)
        self.board.spawn_token_unit(self.remote, Point(2, 1), 1)
        card = U310()
        card.player = self.local
        card.play(Point(2, 2))

        self.assertEqual(target.position, Point(0, 2))
