from card import Card
from enums import TriggerType
from typing import List
from point import Point

class Structure(Card):
    def __init__(self, cost: int, strength: int, triggers: List[TriggerType]=[TriggerType.TURN_START]):
        super().__init__()
        self.cost = cost
        self.strength = strength
        self.triggers = triggers
        self.position: Point = None

    def __eq__(self, other):
        return self.card_id == other.card_id and self.player == other.player and self.position == other.position

    def __repr__(self):
        strength = f"♥{self.strength}{' ' if self.strength < 10 else ''}"

        return f"{self.position} {self.player.order}: {self.card_id} {strength}      "
    
    @property
    def is_at_turn_start(self):
        return TriggerType.TURN_START in self.triggers

    @property
    def is_at_turn_end(self):
        return TriggerType.TURN_END in self.triggers
    
    @property
    def is_on_play(self):
        return TriggerType.ON_PLAY in self.triggers

    def play(self, position: Point):
        self.position = position
        self.player.board.set(self.position, self)

        if self.is_on_play:
            self.activate_ability()

    def deal_damage(self, amount: int):
        self.strength -= amount

        if self.strength <= 0:
            self.destroy()

    def destroy(self):
        self.player.board.set(self.position, None)

    def heal(self, amount: int):
        self.strength += amount