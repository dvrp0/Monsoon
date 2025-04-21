from enums import Faction
from point import Point
from spell import Spell
from target import Target
from test import CardTestCase

class S302(Spell): # Needle Blast
    def __init__(self):
        super().__init__(Faction.NEUTRAL, 6)
        self.ability_targets = 4
        self.ability_damage = 4

    def activate_ability(self, position: Point | None = None):
        targets = self.player.board.get_targets(Target(Target.Kind.ANY, Target.Side.ENEMY), include_base=True)
        self.player.random.shuffle(targets)

        for target in targets[:self.ability_targets]:
            self.player.board.at(target).deal_damage(self.ability_damage, source=self)

class S302Test(CardTestCase):
    def test_ability(self):
        card = S302()
        card.player = self.local
        self.board.spawn_token_unit(self.remote, Point(0, 3), card.ability_damage + 1)
        self.board.spawn_token_unit(self.remote, Point(2, 2), card.ability_damage + 1)
        card.play()

        self.assertEqual(self.board.at(Point(0, 3)).strength, 1)
        self.assertEqual(self.board.at(Point(2, 2)).strength, 1)
        self.assertEqual(self.remote.strength, 20 - card.ability_damage)

        self.board.spawn_token_structure(self.remote, Point(0, 1), card.ability_damage - 1)
        self.board.spawn_token_structure(self.remote, Point(3, 3), card.ability_damage)
        card.play()

        entities = [self.board.at(Point(0, 1)), self.board.at(Point(2, 2)), self.board.at(Point(0, 3)), self.board.at(Point(3, 3))]
        self.assertEqual(entities.count(None), 4 if self.remote.strength == 20 - card.ability_damage else 3)