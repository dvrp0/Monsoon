from card import Card
from target import Target
from point import Point

class Spell(Card):
    def __init__(self, cost: int, targetable: Target = None):
        super().__init__()
        self.cost = cost
        self.targetable: Target = targetable

    def play(self, position: Point | None = None):
        if position is None or position in self.player.board.get_targets(self.targetable):
            self.activate_ability(position)