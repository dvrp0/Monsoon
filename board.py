import random
from enums import UnitType, TriggerType
from unit import Unit
from structure import Structure
from target import Target
from typing import TYPE_CHECKING, List
from point import Point

if TYPE_CHECKING:
    from player import Player

class Board:
    def __init__(self, local: "Player", remote: "Player"):
        self.board = [[None for _ in range(4)] for _ in range(5)]
        self.local = local
        self.remote = remote

        self.local.board = self
        self.remote.board = self

    def __repr__(self):
        rows = []

        for y in range(5):
            rows.append("  ".join(str(self.board[y][x]) if self.board[y][x] is not None else f"({x}, {y}) {'-' * 17}" for x in range(4)))

        return "\n".join([f"{' ' * 46} {self.remote.order}: {self.remote.strength}"] + rows + [f"{' ' * 46} {self.local.order}: {self.local.strength}"])

    def at(self, position: Point):
        return self.board[position.y][position.x]

    def set(self, position: Point, entity: Unit | Structure | None):
        self.board[position.y][position.x] = entity

    def is_ally(self, entity: Unit | Structure):
        return entity.player == self.local

    def clear(self):
        self.board = [[None for _ in range(4)] for _ in range(5)]

    def to_next_turn(self):
        self.local.fill_hand()

        for structure in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_end:
                structure.activate_ability(structure.position)

        temp = self.local
        self.local = self.remote
        self.remote = temp
        print(id(self.local))
        print(id(self.remote))

        self.board = [row[::-1] for row in self.board[::-1]]
        for y in range(5):
            for x in range(4):
                if self.board[y][x] is not None:
                    self.board[y][x].position = Point(x, y)

        for structure in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_start:
                structure.activate_ability(structure.position)

        for unit in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY))]:
            unit.set_path()
            unit.move()

    def get_targets(self, target: Target, exclude: Point=None) -> List[Point]:
        tiles = []

        for y in range(5):
            for x in range(4):
                tile = self.board[y][x]
                is_unit = isinstance(tile, Unit)
                is_structure = isinstance(tile, Structure)

                if is_unit:
                    type_matches = True if target.unit_types is None else any(type in tile.unit_types for type in target.unit_types)

                if not (is_unit or is_structure):
                    continue

                kind_matches = (target.kind == Target.Kind.ANY or
                    (target.kind == Target.Kind.UNIT and is_unit and type_matches) or
                    (target.kind == Target.Kind.STRUCTURE and is_structure))
                side_matches = (target.side == Target.Side.ANY or
                    (target.side == Target.Side.FRIENDLY and self.is_ally(tile)) or
                    (target.side == Target.Side.ENEMY and not self.is_ally(tile)))

                if kind_matches and side_matches:
                    tiles.append(Point(x, y))

        if exclude is not None and exclude in tiles:
            tiles.remove(exclude)

        return tiles

    def get_front_tiles(self, position: Point, target: Target=None) -> List[Point]:
        tiles = [Point(position.x, i) for i in range(position.y - 1, -1, -1)]

        if target is not None:
            targets = self.get_targets(target)
            tiles = [tile for tile in tiles if tile in targets]

        return tiles

    def get_behind_tiles(self, position: Point, target: Target=None) -> List[Point]:
        tiles = [Point(position.x, i) for i in range(position.y + 1, 5)]

        if target is not None:
            targets = self.get_targets(target)
            tiles = [tile for tile in tiles if tile in targets]

        return tiles

    def get_side_tiles(self, position: Point, target: Target=None) -> List[Point]:
        tiles = [
            Point(position.x - 1, position.y),
            Point(position.x + 1, position.y)
        ]

        if target is not None:
            targets = self.get_targets(target)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.x >= 0 and tile.x <= 3 and tile.y >= 0 and tile.y <= 4]

    def get_bordering_tiles(self, position: Point, target: Target=None) -> List[Point]:
        tiles = [
            Point(position.x - 1, position.y),
            Point(position.x + 1, position.y),
            Point(position.x, position.y - 1),
            Point(position.x, position.y + 1)
        ]

        if target is not None:
            targets = self.get_targets(target)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.x >= 0 and tile.x <= 3 and tile.y >= 0 and tile.y <= 4]

    def get_surrounding_tiles(self, position: Point, target: Target=None) -> List[Point]:
        tiles = [
            Point(position.x - 1, position.y),
            Point(position.x - 1, position.y - 1),
            Point(position.x - 1, position.y + 1),
            Point(position.x + 1, position.y),
            Point(position.x + 1, position.y - 1),
            Point(position.x + 1, position.y + 1),
            Point(position.x, position.y - 1),
            Point(position.x, position.y + 1)
        ]

        if target is not None:
            targets = self.get_targets(target)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.x >= 0 and tile.x <= 3 and tile.y >= 0 and tile.y <= 4]

    def spawn_token_unit(self, player: "Player", position: Point, strength: int, types: List[UnitType] = None):
        types = types or [random.choice(list(UnitType))]

        token = Unit(types, 0, strength, 1)
        token.player = player
        token.position = position

        self.set(position, token)

    def spawn_token_structure(self, player: "Player", position: Point, strength: int):
        token = Structure(0, strength)
        token.player = player
        token.position = position

        self.set(position, token)