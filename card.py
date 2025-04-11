from abc import ABC, abstractmethod
from copy import deepcopy
from enums import Faction
from typing import TYPE_CHECKING
from point import Point
import uuid

if TYPE_CHECKING:
    from player import Player

class Card(ABC):
    def __init__(self):
        self.uuid = uuid.uuid4()
        self.card_id = self.__class__.__name__.lower()[:4]
        self.player: Player = None
        self.faction: Faction = Faction.NEUTRAL
        self.cost = 0
        self.weight = 0
        self.is_single_use = False

    def __eq__(self, other):
        return self.uuid == other.uuid if isinstance(other, Card) else False

    def __int__(self):
        if self.card_id[:2] == "cs":
            kind = "3"
        else:
            match self.card_id[0]:
                case "b":
                    kind = "0"
                case "s":
                    kind = "1"
                case "u":
                    kind = "2"
                case "t":
                    kind = "3"
                case "f":
                    kind = "4"

        return int(f"{kind}{self.card_id[1:]}", 16)

    @abstractmethod
    def play(self, position: Point | None = None):
        pass

    def activate_ability(self, position: Point | None = None):
        pass

    def copy(self):
        copied = deepcopy(self)
        copied.player = self.player # 그냥 deepcopy()만 하면 player가 새로운 객체가 되어서 난리가 남

        return copied