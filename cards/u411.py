from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context, Target
from test import CardTestCase

class U411(Unit): # Copperskin Rangers
    def __init__(self):
        super().__init__(Faction.SHADOWFEN, [UnitType.TOAD], 2, 3, 0, TriggerType.ON_PLAY)
        self.ability_amount = 3

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        targets = self.player.board.get_targets(
            Context(source=self),
            Target(Target.Kind.UNIT, Target.Side.ENEMY)
        )

        if len(targets) > 0:
            self.player.random.shuffle(targets)

            for target in targets[:self.ability_amount]:
                self.player.board.at(target).poison()

class U411Test(CardTestCase):
    def test_ability(self):
        r1 = self.board.spawn_token_unit(self.remote, Point(3, 1), 3)
        r2 = self.board.spawn_token_unit(self.remote, Point(1, 3), 3)
        r3 = self.board.spawn_token_unit(self.remote, Point(1, 0), 3)
        l = self.board.spawn_token_unit(self.local, Point(3, 4), 3)
        card = U411()
        card.player = self.local
        card.play(Point(0, 4))

        self.assertTrue(r1.is_poisoned or r2.is_poisoned or r3.is_poisoned)
        self.assertFalse(l.is_poisoned)