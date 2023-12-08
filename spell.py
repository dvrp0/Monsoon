from card import Card
from target import Target
from point import Point

class Spell(Card):
    def __init__(self, cost: int, required_targets: Target | None = None):
        super().__init__()
        self.cost = cost
        self.required_targets = required_targets

    def play(self, position: Point | None = None):
        if self.required_targets is None or position in self.player.board.get_targets(self.required_targets):
            self.activate_ability(position)