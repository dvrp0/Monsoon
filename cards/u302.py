from enums import UnitType, TriggerType
from point import Point
from unit import Unit
from test import CardTestCase

class U302(Unit): # Windmakers
    def __init__(self):
        super().__init__([UnitType.RODENT], 4, 2, 2, TriggerType.BEFORE_ATTACKING)

    def activate_ability(self, position: Point | None = None):
        target = self.player.board.at(position)

        if target.strength > self.strength:
            target.deal_damage(7)
            target.push(self.position)

class U302Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 10)
        card = U302()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(self.board.at(Point(0, 1)).strength, 3)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 3), 10)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 8)
        card = U302()
        card.player = self.local
        card.play(Point(0, 3))

        self.assertEqual(card.position, Point(1, 2))
        self.assertEqual(self.board.at(Point(1, 0)).strength, 1)
        self.assertEqual(self.board.at(Point(3, 3)).strength, 3)