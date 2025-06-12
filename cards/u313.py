from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U313(Unit): # Debug Loggers
    def __init__(self):
        super().__init__(Faction.IRONCLAD, [UnitType.CONSUTRUCT], 5, 8, 1, TriggerType.AFTER_ATTACKING)
        self.ability_strength = 5

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_surrounding_tiles(
            Context(self.position, source=self),
            Target(Target.Kind.UNIT, Target.Side.FRIENDLY)
        )

        if len(targets) > 0:
            self.player.board.at(self.player.random.choice(targets)).heal(self.ability_strength)

        self.heal(self.ability_strength)

class U313Test(CardTestCase):
    def test_ability(self):
        card = U313()
        card.player = self.local
        u1 = self.board.spawn_token_unit(self.local, Point(1, 4), 1)
        card.play(Point(0, 4))