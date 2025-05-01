from card import Card
from enums import Faction, UnitType, TriggerType
from point import Point
from unit import Unit
from target import Context, Target
from test import CardTestCase

class UE04(Unit): # Greenwood Ancients
    def __init__(self):
        super().__init__(Faction.NEUTRAL, [UnitType.ELDER], 4, 6, 1, TriggerType.AFTER_SURVIVING)
        self.ability_strength = 4

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(pov=self.player, source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )
        self.heal(sum(self.player.board.at(target).strength > self.strength for target in targets) * self.ability_strength)

class UE04Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 0), 1)
        self.board.spawn_token_unit(self.remote, Point(1, 0), 2)
        self.board.spawn_token_unit(self.remote, Point(2, 0), 7)
        self.board.spawn_token_unit(self.remote, Point(3, 0), 12)
        self.board.spawn_token_unit(self.remote, Point(0, 1), 7)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 1)
        card = UE04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, UE04().strength - 1 + card.ability_strength * 3)

        self.board.clear()
        self.board.spawn_token_unit(self.remote, Point(0, 0), 3)
        self.board.spawn_token_unit(self.remote, Point(0, 3), 3)
        card = UE04()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertEqual(card.strength, 3)