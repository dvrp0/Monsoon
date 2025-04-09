from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from test import CardTestCase

class U302(Unit): # Windmakers
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.RODENT], 4, 2, 2, TriggerType.BEFORE_ATTACKING)

    def activate_ability(self, position: Point | None = None):
        if not position.is_valid or not isinstance(self.player.board.at(position), Unit): # 기지 또는 건물이라면 발동 안 함
            return

        target = self.player.board.at(position)

        if target.strength > self.strength:
            target.deal_damage(7)

            if target.strength > 0:
                target.push(self.position)

class U302Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_structure(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 3)
        card = U302()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.position, Point(0, 2))
        self.assertEqual(self.board.at(Point(0, 1)), None)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(1, 3), 10)
        self.board.spawn_token_unit(self.remote, Point(1, 2), 8)
        card = U302()
        card.player = self.local
        card.play(Point(0, 3))

        self.assertEqual(card.position, Point(1, 2))
        self.assertEqual(self.board.at(Point(1, 0)).strength, 1)
        self.assertEqual(self.board.at(Point(3, 3)).strength, 3)

        self.board.clear()
        card = U302()
        card.player = self.local
        card.play(Point(0, 1))

        self.assertEqual(self.board.at(Point(0, 0)), None)
        self.assertEqual(self.remote.strength, 18)

        self.board.spawn_token_structure(self.remote, Point(1, 2), 10)
        card = U302()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.board.at(Point(1, 2)).strength, 8)