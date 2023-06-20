from abc import ABC, abstractmethod
from point import Point

class Card(ABC):
    def __init__(self):
        self.card_id = self.__class__.__name__.lower()[:4]
        self.cost = 0
        self.weight = 0
        self.is_single_use = False

    @abstractmethod
    def play(self, position: Point | None=None):
        pass

    def activate_ability(self, position: Point | None=None):
        pass