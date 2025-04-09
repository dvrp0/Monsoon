from enums import Faction
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S302(Spell): # Needle Blast
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 6)

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY)) + [Point(-1, -1)] # indicating enemy base
        self.player.random.shuffle(targets)

        for target in targets[:4]:
            if target == Point(-1, -1):
                self.player.opponent.deal_damage(4)
            else:
                self.player.board.at(target).deal_damage(4)

class S302Test(CardTestCase):
    def test_ability(self):
        self.board.spawn_token_unit(self.remote, Point(0, 3), 5)
        self.board.spawn_token_unit(self.remote, Point(2, 2), 5)
        card = S302()
        card.player = self.local
        card.play()

        self.assertEqual(self.board.at(Point(0, 3)).strength, 1)
        self.assertEqual(self.board.at(Point(2, 2)).strength, 1)
        self.assertEqual(self.remote.strength, 16)

        self.board.spawn_token_structure(self.remote, Point(0, 1), 3)
        self.board.spawn_token_structure(self.remote, Point(3, 3), 4)
        card.play()

        entities = [self.board.at(Point(0, 1)), self.board.at(Point(2, 2)), self.board.at(Point(0, 3)), self.board.at(Point(3, 3))]
        self.assertEqual(entities.count(None), 4 if self.remote.strength == 16 else 3)