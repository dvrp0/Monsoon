from enum import IntEnum
from enums import UnitType, StatusEffect
from typing import List
from card import Card
from point import Point

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
        status_effects: List[StatusEffect] = None, exclude_status_effects: List[StatusEffect] = None,
        include_base: bool = False):
        self.kind = kind
        self.side = side
        self.unit_types = unit_types
        self.exclude_unit_types = exclude_unit_types
        self.strength_limit = strength_limit
        self.non_hero = non_hero
        self.status_effects = status_effects
        self.exclude_status_effects = exclude_status_effects
        self.include_base = include_base

class Context:
    def __init__(self, position: Point | None = None, exclude: Point | None = None,
        pov: Card | None = None, source: Card | None = None, natural=False):
        # natural: if False, default perspective is that of non-attacking player
        self.position = position
        self.exclude = exclude
        self.pov = pov
        self.source = source
        self.natural = natural