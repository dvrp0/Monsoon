from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from point import Point

if TYPE_CHECKING:
    from player import Player

class Card(ABC):
    def __init__(self):
        self.card_id = self.__class__.__name__.lower()[:4]
        self.player: Player = None
        self.cost = 0
        self.weight = 0
        self.is_single_use = False

    @abstractmethod
    def play(self, position: Point | None = None):
        pass

    def activate_ability(self, position: Point | None = None):
        pass

    def copy(self):
        copied = deepcopy(self)
        copied.player = self.player # 그냥 deepcopy()만 하면 player가 새로운 객체가 되어서 난리가 남

        return copied