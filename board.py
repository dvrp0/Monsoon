import numpy as np
from enums import UnitType
from card import Card
from unit import Unit
from structure import Structure
from target import Target
from typing import TYPE_CHECKING, List
from point import Point

if TYPE_CHECKING:
    from player import Player

class Board:
    def __init__(self, local: "Player", remote: "Player", random: np.random.RandomState):
        self.board: List[List[Unit | Structure | None]] = [[None for _ in range(4)] for _ in range(5)]
        self.local = local
        self.remote = remote
        self.history: List[Card] = []
        self.random = random

        self.local.board = self
        self.remote.board = self

    def __repr__(self):
        rows = []

        for y in range(5):
            row = "  ".join(str(self.board[y][x]) if self.board[y][x] is not None else f"({x}, {y}) {'-' * 17}" for x in range(4))
            rows.append(f"{'▶' if self.local.front_line == y else ' '} {row} {'◁' if self.remote.front_line == y else ' '}")

        remote = [f"{' ' * 50} {self.remote.order}: {self.remote.strength}"]
        local = [f"{' ' * 50} {self.local.order}: {self.local.strength}"]
        history = [", ".join(f"[{'local' if card.player == self.local else 'remote'}: {card.card_id}]" for card in self.history[-4:])]

        return "\n".join(remote + rows + local + history)

    def at(self, position: Point):
        return self.board[position.y][position.x]

    def set(self, position: Point, entity: Unit | Structure | None):
        self.board[position.y][position.x] = entity

    def is_ally(self, entity: Unit | Structure):
        return entity.player == self.local

    def clear(self):
        self.board = [[None for _ in range(4)] for _ in range(5)]
        self.local.front_line = 4
        self.remote.front_line = 0

    def calculate_front_line(self, player: "Player"):
        if player == self.local:
            for y in range(5):
                if any(self.board[y][x] is not None and self.board[y][x].player == player for x in range(4)):
                    self.local.front_line = max(1, y) # farthest front line is 1
                    break

                self.local.front_line = 4
        elif player == self.remote:
            for y in range(4, -1, -1):
                if any(self.board[y][x] is not None and self.board[y][x].player == player for x in range(4)):
                    self.remote.front_line = min(3, y) # farthest front line is 3
                    break

                self.remote.front_line = 0

    def flip(self):
        temp = self.local
        self.local = self.remote
        self.remote = temp

        self.local.front_line = 4 - self.local.front_line
        self.remote.front_line = 4 - self.remote.front_line

        self.board = [row[::-1] for row in self.board[::-1]]
        for y in range(5):
            for x in range(4):
                if self.board[y][x] is not None:
                    self.board[y][x].position = Point(x, y)

    def to_next_turn(self):
        self.local.fill_hand()

        for structure in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_end:
                structure.activate_ability(structure.position)

        self.calculate_front_line(self.local)
        self.calculate_front_line(self.remote)

        self.local.max_mana += 1
        self.local.current_mana = self.local.max_mana
        self.remote.current_mana = self.remote.max_mana

        self.flip()

        self.local.replacable = True
        self.local.leftmost_movable = True

        for structure in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.STRUCTURE, Target.Side.FRIENDLY))]:
            if structure.is_at_turn_start:
                structure.activate_ability(structure.position)

        for unit in [self.at(tile) for tile in self.get_targets(Target(Target.Kind.UNIT, Target.Side.FRIENDLY))]:
            unit.set_path()
            unit.move()

    def get_targets(self, target: Target, exclude: Point = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
        tiles = []

        for y in range(5):
            for x in range(4):
                entity = self.board[y][x]

                if entity is None or entity.strength <= 0:
                    continue

                is_unit = isinstance(entity, Unit)
                is_structure = isinstance(entity, Structure)

                if is_unit:
                    type_matches = True if target.unit_types is None else any(type in entity.unit_types for type in target.unit_types)
                    non_hero_matches = True if not target.non_hero else UnitType.HERO not in entity.unit_types
                    status_matches = True if target.status_effects is None else any(status in entity.status_effects for status in target.status_effects)

                strength_matches = True if target.strength_limit is None else entity.strength <= target.strength_limit
                unit_matches = is_unit and type_matches and non_hero_matches and status_matches and strength_matches
                structure_matches = is_structure and strength_matches

                kind_matches = ((target.kind == Target.Kind.ANY and (unit_matches or structure_matches)) or
                    (target.kind == Target.Kind.UNIT and unit_matches) or
                    (target.kind == Target.Kind.STRUCTURE and structure_matches))
                side_matches = (target.side == Target.Side.ANY or
                    (target.side == Target.Side.FRIENDLY and entity.player == perspective) or
                    (target.side == Target.Side.ENEMY and entity.player != perspective))

                if kind_matches and side_matches:
                    tiles.append(Point(x, y))

        if exclude is not None and exclude in tiles:
            tiles.remove(exclude)

        return tiles

    def get_front_tiles(self, position: Point, target: Target = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
        tiles = [Point(position.x, i) for i in range(position.y - 1, -1, -1)]

        if target is not None:
            targets = self.get_targets(target, perspective=perspective)
            tiles = [tile for tile in tiles if tile in targets]

        return tiles

    def get_behind_tiles(self, position: Point, target: Target = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
        tiles = [Point(position.x, i) for i in range(position.y + 1, 5)]

        if target is not None:
            targets = self.get_targets(target, perspective=perspective)
            tiles = [tile for tile in tiles if tile in targets]

        return tiles

    def get_side_tiles(self, position: Point, target: Target = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
        tiles = [
            Point(position.x - 1, position.y),
            Point(position.x + 1, position.y)
        ]

        if target is not None:
            targets = self.get_targets(target, perspective=perspective)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.is_valid]

    def get_bordering_tiles(self, position: Point, target: Target = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
        tiles = [
            Point(position.x - 1, position.y),
            Point(position.x + 1, position.y),
            Point(position.x, position.y - 1),
            Point(position.x, position.y + 1)
        ]

        if target is not None:
            targets = self.get_targets(target, perspective=perspective)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.is_valid]

    def get_surrounding_tiles(self, position: Point, target: Target = None, perspective: "Player | None" = None) -> List[Point]:
        if perspective is None:
            perspective = self.local
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
            targets = self.get_targets(target, perspective=perspective)
            tiles = [tile for tile in tiles if tile in targets]

        return [tile for tile in tiles if tile.is_valid]

    def spawn_token_unit(self, player: "Player", position: Point, strength: int, types: List[UnitType] = None):
        types = types or [UnitType(self.random.choice(list(UnitType)))]

        token = Unit(types, 0, strength, 1)
        token.player = player
        token.position = position

        types_id = "".join([str(type.value) for type in types])
        token.card_id = f"f{types_id.zfill(3)}"

        self.set(position, token)
        self.calculate_front_line(player)

    def spawn_token_structure(self, player: "Player", position: Point, strength: int):
        token = Structure(0, strength)
        token.player = player
        token.position = position
        token.card_id = "b001"

        self.set(position, token)
        self.calculate_front_line(player)

    def add_to_history(self, card: Card):
        self.history.append(card)