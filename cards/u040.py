from card import Card
from enums import Faction, UnitType, TriggerType
from unit import Unit
from point import Point
from target import Context
from test import CardTestCase

class U040(Unit): # Diehards
    def __init__(self):
        super().__init__(Faction.SWARM, [UnitType.UNDEAD], 3, 6, 0, TriggerType.ON_DEATH)
        self.ability_strength = 12

    def activate_ability(self, position: Point | None = None, source: Card | None = None):
        print(source)
        if source is not None:
            tiles = self.player.board.get_surrounding_tiles(
                Context(self.position, pov=self.player, source=self)
            )

            if len(tiles) > 0:
                # FIXME: Does respawning copy mana cost?
                self.respawn(self.player.random.choice(tiles), self.ability_strength)

class U040Test(CardTestCase):
    def test_ability(self):
        from cards.s001 import S001

        card = U040()
        card.player = self.local
        card.play(Point(0, 4))
        self.board.to_next_turn()
        s001 = S001()
        s001.player = self.remote
        s001.play(Point(0, 4))

        self.assertTrue(self.board.at(Point(0, 3)) is not None or \
            self.board.at(Point(1, 3)) is not None or self.board.at(Point(1, 4)) is not None)

        self.board.clear()
        card = U040()
        card.player = self.local
        card.play(Point(0, 4))
        u1 = self.board.spawn_token_unit(self.remote, Point(0, 3), card.ability_strength)
        self.board.to_next_turn()

        self.assertEqual(u1.strength, card.ability_strength - U040().strength)