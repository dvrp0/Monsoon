from enum import IntEnum
from enums import UnitType, StatusEffect
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

    def __init__(self, kind: Kind, side: Side, unit_types: List[UnitType] = None, exclude_unit_types: List[UnitType] = None,
        strength_limit: int = None, non_hero: bool = False,
        status_effects: List[StatusEffect] = None, exclude_status_effects: List[StatusEffect] = None):
        self.kind = kind
        self.side = side
        self.unit_types = unit_types
        self.exclude_unit_types = exclude_unit_types
        self.strength_limit = strength_limit
        self.non_hero = non_hero
        self.status_effects = status_effects
        self.exclude_status_effects = exclude_status_effects