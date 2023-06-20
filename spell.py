from card import Card
from target import Target

class Spell(Card):
    def __init__(self, cost: int):
        super().__init__()
        self.cost = cost
        self.targetable: Target = None