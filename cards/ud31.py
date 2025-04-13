from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Target
from test import CardTestCase

class UD31(Unit): # Greengale Surpents
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.DRAGON], 3, 4, 2, TriggerType.BEFORE_ATTACKING)
        self.ability_strength = 2

    def activate_ability(self, position: Point | None = None):
        if not position.is_valid or not isinstance(self.player.board.at(position), Unit): # 기지 또는 건물이라면 발동 안 함
            return

        targets = self.player.board.get_surrounding_tiles(self.position, Target(Target.Kind.UNIT, Target.Side.FRIENDLY, [UnitType.DRAGON]), self.player)

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)

        self.heal(self.ability_strength)

class UD31Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.local, Point(0, 3), 1, [UnitType.DRAGON])
        self.board.spawn_token_unit(self.remote, Point(1, 4), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 3), 1)
        card = UD31()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, UD31().strength - 2 + card.ability_strength * 2)
        self.assertEqual(self.board.at(Point(0, 3)).strength, 1 + card.ability_strength * 2)

        self.board.clear()
        card = UD31()
        card.player = self.local
        card.play(Point(0, 1))

        self.assertEqual(self.board.at(Point(0, 0)), None)
        self.assertEqual(self.remote.strength, 20 - UD31().strength)

        self.board.spawn_token_structure(self.remote, Point(1, 2), 10)
        card = UD31()
        card.player = self.local
        card.play(Point(1, 3))

        self.assertEqual(self.board.at(Point(1, 2)).strength, 10 - UD31().strength)