from enum import IntEnum
from enums import UnitType
from typing import List

class Target:
    class Kind(IntEnum):
        UNIT = 0
        STRUCTURE = 1
        ANY = 2

    class Side(IntEnum):
        FRIENDLY = 0
        ENEMY = 1
        ANY = 2

    def __init__(self, kind: Kind, side: Side, unit_types: List[UnitType]=None):
        self.kind = kind
        self.side = side
        self.unit_types = unit_types